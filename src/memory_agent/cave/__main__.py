import argparse
import json
import math
import os
import random
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from . import mini
from .models import FrozenQwen, SYSTEM, teacher_request, xml_parser
from .records import digest, read, save, verify


def collect_episode(root, actor, parser, spec, episode_id, seed, max_steps, fixed_opportunity=None):
    state = mini.initial(spec)
    prompt = spec["instruction"] + "\nCurrent working list: " + str(state["working"])
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    # The checkpoint draw and per-turn generation seeds are independent streams.
    main_step = fixed_opportunity or random.Random("checkpoint:" + str(seed)).randint(1, max_steps)
    q = 1.0 if fixed_opportunity is not None else 1/max_steps
    episode = root / episode_id
    episode.mkdir()
    end, steps, tool_calls, opportunities = "budget", [], 0, 0
    input_tokens = output_tokens = 0
    save(episode / "sampling.json", {"main_opportunity": main_step, "q": q, "seed": seed})
    for step in range(1, max_steps + 1):
        checkpoint = f"{episode_id}/{step:03d}"
        folder = root / checkpoint
        decode_seed = random.Random(f"actor:{seed}:{step}").randrange(2**31)
        pre = {"checkpoint_id": checkpoint, "step": step, "messages": messages, "state": state,
               "decode_seed": decode_seed, "remaining_steps": max_steps-step+1,
               "used_input_tokens": input_tokens, "used_output_tokens": output_tokens,
               "used_tool_calls": tool_calls, "memory_enabled": False, "teacher_enabled": False}
        save(folder / "pre.json", pre)
        try:
            generation = actor.generate(messages, decode_seed)
        except Exception as exc:
            save(folder / "generation_error.json", {"status": "error", "error_type": type(exc).__name__})
            steps.append({"step": step, "turn_id": step, "opportunity_index": None,
                          "checkpoint_id": checkpoint, "executable": False, "status": "actor_error"})
            end = "actor_error"
            break
        save(folder / "generation.json", generation)
        parsed = parser.parse_response(generation["text"])
        batch = [asdict(c) for c in parsed.commands]
        feedback = parsed.error
        if any(not math.isfinite(cmd["duration"]) for cmd in batch):
            feedback = "nonfinite duration"
            batch = []
        action = {"commands": batch, "complete": parsed.is_task_complete}
        messages = messages + [{"role": "assistant", "content": generation["text"]}]
        input_tokens += generation["prompt_tokens"]
        output_tokens += generation["completion_tokens"]
        save(folder / "draft.json", {"checkpoint_id": checkpoint, "messages": messages, "state": state,
             "a0": action, "a0_digest": digest(action), "error": feedback, "warning": parsed.warning,
             "remaining_steps": max_steps-step, "used_input_tokens": input_tokens,
             "used_output_tokens": output_tokens, "used_tool_calls": tool_calls})
        executable = not feedback and bool(batch)
        if executable:
            opportunities += 1
            cached = read(folder / "draft.json")["a0"]
            obs = mini.execute(state, cached["commands"])
            tool_calls += len(cached["commands"])
            executed = cached
        else:
            obs, executed = [feedback or "no executable command; provide a keystrokes tag"], None
        save(folder / "execution.json", {"executed_action": executed, "observation": obs, "state": state,
                                         "used_tool_calls": tool_calls})
        steps.append({"step": step, "turn_id": step,
                      "opportunity_index": opportunities if executable else None,
                      "checkpoint_id": checkpoint, "executable": executable})
        if not feedback and (state["submitted"] is not None or parsed.is_task_complete):
            end = "submitted" if state["submitted"] is not None else "actor_complete"
            break
        messages = messages + [{"role": "user", "content": "\n".join(obs)}]
    score = ({"status": "unavailable", "reward": None, "error_type": "actor_error"}
             if end == "actor_error" else verify(spec, state))
    save(episode / "verdict.json", score)
    main = next((s for s in steps if s["opportunity_index"] == main_step), None)
    return {"episode_id": episode_id, "task": spec, "seed": seed, "steps": steps,
            "main_opportunity": main_step, "main_reached": main is not None,
            "main_turn_id": main["turn_id"] if main else None,
            "main_status": "eligible" if main else "never_reached",
            "q": q, "termination": end, "verifier": score,
            "input_tokens": input_tokens, "output_tokens": output_tokens, "tool_calls": tool_calls}


def collect(args):
    if args.episodes < 1 or args.max_steps < 1 or args.max_new_tokens < 1 or args.temperature < 0:
        raise ValueError("invalid collection budget or temperature")
    actor = FrozenQwen(args.model, args.max_new_tokens, args.temperature)
    parser = xml_parser()
    run = args.out
    run.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    tasks = [mini.task(args.split, args.seed + n) for n in range(args.episodes)]
    manifest = {"schema": 1, "run_id": run_id, "benchmark": "mini-data-v1", "split": args.split,
                "split_plan": mini.TEMPLATES, "tasks": tasks, "actor": actor.info,
                "parser": type(parser).__name__, "max_steps": args.max_steps,
                "seed": args.seed, "memory_enabled": False, "teacher_enabled": False}
    save(run / "manifest.json", manifest)
    for n, spec in enumerate(tasks):
        item = collect_episode(run, actor, parser, spec, f"ep-{n:04d}", args.seed+n, args.max_steps)
        item["run_id"] = run_id
        item["split"] = args.split
        with (run / "index.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item) + "\n")
        print(item["episode_id"], item["termination"], item["verifier"]["status"], flush=True)


def propose(args):
    run = args.run
    manifest = read(run / "manifest.json")
    if manifest["split"] != "source":
        raise ValueError("only source runs can be sent to the teacher")
    episodes = [json.loads(line) for line in (run / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    item = next((v for v in episodes if v["split"] == "source" and v["run_id"] == manifest["run_id"]
                 and v["verifier"]["status"] == "fail"
                 and (args.episode is None or v["episode_id"] == args.episode)), None)
    if item is None:
        raise ValueError("no independently scored failed source episode matches the selection")
    steps = []
    for node in item["steps"]:
        folder = run / node["checkpoint_id"]
        draft = read(folder / "draft.json")
        execution = read(folder / "execution.json")
        steps.append({"step": node["step"], "a0": draft["a0"], "error": draft["error"],
                      "executed": execution["executed_action"], "observation": execution["observation"]})
    source = {"run_id": item["run_id"], "episode_id": item["episode_id"], "task": item["task"],
              "steps": steps, "score": item["verifier"]}
    output = run / "proposals" / (item["episode_id"] + "-" + uuid.uuid4().hex + ".json")
    start = time.monotonic()
    try:
        result = teacher_request(source, args.base_url, args.model, args.api_style)
    except Exception as exc:
        result = {"status": "error", "error_type": type(exc).__name__, "model": args.model,
                  "seconds": time.monotonic()-start, "http_status": None, "usage": None,
                  **getattr(exc, "audit", {}), "source_digest": digest(source)}
        save(output, {**result, "run_id": item["run_id"], "episode_id": item["episode_id"]})
        raise
    save(output, {**result, "status": "ok", "run_id": item["run_id"], "episode_id": item["episode_id"]})
    print(output)


def main():
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-7B")
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--split", choices=list(mini.TEMPLATES), required=True)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--episodes", type=int, default=1)
    c.add_argument("--max-steps", type=int, default=4)
    c.add_argument("--max-new-tokens", type=int, default=384)
    c.add_argument("--temperature", type=float, default=0.0)
    p = sub.add_parser("propose")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--episode")
    p.add_argument("--model", default="gpt-6-sol")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--api-style", choices=["chat", "responses"], default="chat")
    v = sub.add_parser("verifier-smoke")
    v.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.command == "collect":
        collect(args)
    elif args.command == "propose":
        propose(args)
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        spec = mini.task("source", 42)
        state = mini.initial(spec)
        for label, submitted in [("pass", sorted(spec["input"])), ("fail", []), ("unavailable", None)]:
            state["submitted"] = submitted
            record = verify(spec, state, args.out / "absent.py" if label == "unavailable" else None)
            save(args.out / (label + ".json"), record)
            if record["status"] != label:
                raise RuntimeError(f"verifier-smoke {label} mismatch")
        print(args.out)


if __name__ == "__main__":
    main()
