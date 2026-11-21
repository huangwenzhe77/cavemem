"""Independent, deterministic replay of archived mini-data trajectories."""

import json
import math
import random
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from . import mini
from .models import SYSTEM, xml_parser
from .records import digest, read, verify


def require(ok, message):
    if not ok:
        raise ValueError(message)


def protocol_digest():
    base = Path(__file__).parent
    import hashlib

    return digest(
        {
            n: hashlib.sha256((base / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for n in (
                "mini.py",
                "verifier.py",
                "replay.py",
                "branches.py",
                "experiment.py",
                "models.py",
                "reviewed_contracts.json",
            )
        }
    )


def load_run(run):
    manifest = read(run / "manifest.json")
    require(
        manifest["benchmark"] == "mini-data-v1"
        and not manifest["memory_enabled"]
        and not manifest["teacher_enabled"],
        "expected a no-memory mini-data run",
    )
    rows = [json.loads(s) for s in (run / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    require([r["task"] for r in rows] == manifest["tasks"], "incomplete or reordered task ledger")
    for n, row in enumerate(rows):
        require(
            row["run_id"] == manifest["run_id"]
            and row["split"] == manifest["split"]
            and row["episode_id"] == f"ep-{n:04d}",
            "episode lineage mismatch",
        )
    return manifest, rows


def parse(text, parser):
    result = parser.parse_response(text)
    batch = [asdict(c) for c in result.commands]
    error = result.error
    if any(not math.isfinite(cmd["duration"]) for cmd in batch):
        batch, error = [], "nonfinite duration"
    return {"commands": batch, "complete": result.is_task_complete}, error


def advance(state, action, error):
    if not error and action["commands"]:
        observation = mini.execute(state, action["commands"])
        executed = action
    else:
        observation = [error or "no executable command; provide a keystrokes tag"]
        executed = None
    end = None
    if not error and state["submitted"] is not None:
        end = "submitted"
    elif not error and action["complete"]:
        end = "actor_complete"
    return observation, executed, end


def nodes(run, item):
    for i, step in enumerate(item["steps"], start=1):
        path = f"{item['episode_id']}/{i:03d}"
        require(step["checkpoint_id"] == path and step["turn_id"] == i, "bad checkpoint identity")
        yield i, run / path, step


def restore(run, item, turn):
    """Decode new objects for each fork. Callers cannot share a mutable session."""
    folder = run / item["episode_id"] / f"{turn:03d}"
    return tuple(read(folder / name) for name in ("pre.json", "draft.json", "generation.json"))


def replay_tape(run, manifest, item, start=1, replacement=None):
    parser = xml_parser()
    spec = item["task"]
    state = mini.initial(spec)
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": spec["instruction"] + "\nCurrent working list: " + str(state["working"]),
        },
    ]
    used_in = used_out = tools = opportunities = 0
    logs = []
    termination = "budget"
    for turn, folder, node in nodes(run, item):
        pre, draft, generation = restore(run, item, turn)
        check = f"{item['episode_id']} turn {turn}"
        require(pre["checkpoint_id"] == node["checkpoint_id"] == draft["checkpoint_id"], check + " ID")
        require(
            pre["step"] == turn
            and pre["remaining_steps"] == manifest["max_steps"] - turn + 1
            and pre["decode_seed"] == random.Random(f"actor:{item['seed']}:{turn}").randrange(2**31)
            and not pre["memory_enabled"]
            and not pre["teacher_enabled"],
            check + " boundary",
        )
        require(pre["state"] == state == draft["state"] and pre["messages"] == messages, check + " Hpre")
        if turn == start:
            state, messages = deepcopy(pre["state"]), deepcopy(pre["messages"])
        require(
            (used_in, used_out, tools)
            == (pre["used_input_tokens"], pre["used_output_tokens"], pre["used_tool_calls"]),
            check + " pre budget",
        )
        text = generation["text"]
        action, error = parse(text, parser)
        require(
            action == draft["a0"] and digest(action) == draft["a0_digest"] and error == draft["error"],
            check + " cached draft",
        )
        require(
            generation["status"] == "ok" and generation["seed"] == pre["decode_seed"],
            check + " generation",
        )
        prompt = (
            "\n\n".join(m["role"].upper() + ":\n" + m["content"] for m in messages) + "\n\nASSISTANT:\n"
        )
        require(generation["prompt_digest"] == digest(prompt), check + " input digest")
        messages.append({"role": "assistant", "content": text})
        used_in += generation["prompt_tokens"]
        used_out += generation["completion_tokens"]
        require(
            messages == draft["messages"]
            and draft["remaining_steps"] == pre["remaining_steps"] - 1
            and (used_in, used_out, tools)
            == (draft["used_input_tokens"], draft["used_output_tokens"], draft["used_tool_calls"]),
            check + " Hpost/budget",
        )
        if turn == start and replacement is not None:
            action, error = replacement, ""
        obs, executed, end = advance(state, action, error)
        tools += len(executed["commands"]) if executed else 0
        row = {
            "turn": turn,
            "state": deepcopy(state),
            "observation": obs,
            "executed_action": executed,
            "used_tool_calls": tools,
            "used_input_tokens": used_in,
            "used_output_tokens": used_out,
            "session_digest_before_action": digest(messages),
            "remaining_turns": draft["remaining_steps"],
        }
        logs.append(row)
        if replacement is not None and turn == start:
            # A nonterminal repair would require a fresh actor-conditioned suffix.
            require(end is not None, "nonterminal repair needs independent actor continuations")
        else:
            archived = read(folder / "execution.json")
            require(
                all(
                    row[k] == archived[k]
                    for k in ("state", "observation", "executed_action", "used_tool_calls")
                ),
                check + " execution",
            )
            if executed:
                opportunities += 1
            require(
                node["opportunity_index"] == (opportunities if executed else None)
                and node["executable"] == bool(executed),
                check + " opportunity",
            )
        if end:
            termination = end
            break
        messages.append({"role": "user", "content": "\n".join(obs)})
    score = verify(spec, state)
    require(score["status"] in ("pass", "fail"), "independent verifier unavailable")
    if replacement is None:
        require(
            len(logs) == len(item["steps"])
            and (termination != "budget" or len(logs) == manifest["max_steps"]),
            "truncated trajectory",
        )
        require(
            (item["input_tokens"], item["output_tokens"], item["tool_calls"])
            == (used_in, used_out, tools),
            "final budget mismatch",
        )
        original = read(run / item["episode_id"] / "verdict.json")
        require(
            item["termination"] == termination
            and original == item["verifier"]
            and original["reward"] == score["reward"]
            and original["status"] == score["status"],
            "original terminal verdict mismatch",
        )
    return {
        "turns": logs,
        "score": score,
        "termination": termination,
        "final_state": state,
        "trace_digest": digest(logs),
    }


def selected_turn(run, manifest, item):
    horizon = manifest["max_steps"]
    fixed = manifest.get("cohort") == "controlled_fixed_prefix"
    j = 2 if fixed else random.Random("checkpoint:" + str(item["seed"])).randint(1, horizon)
    q = 1.0 if fixed else 1 / horizon
    sampling = read(run / item["episode_id"] / "sampling.json")
    require(
        j == sampling["main_opportunity"] == item["main_opportunity"]
        and sampling["q"] == item["q"] == q
        and sampling["seed"] == item["seed"],
        "opportunity schedule changed",
    )
    turn = next((s["turn_id"] for s in item["steps"] if s["opportunity_index"] == j), None)
    require(turn == item["main_turn_id"] and bool(turn) == item["main_reached"], "index changed")
    return turn
