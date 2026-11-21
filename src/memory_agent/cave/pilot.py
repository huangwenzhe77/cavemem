"""Card 07 driver; consumes the frozen card 04–06 bank without changing its protocol."""

import argparse
import json
import random
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from . import mini
from .__main__ import collect_episode
from .experiment import audit_aa, match, open_bank, pair
from .models import FrozenQwen, xml_parser
from .records import digest, read, save
from .replay import load_run, parse, protocol_digest, require, restore, selected_turn


STRATA = (("controlled_transfer", 71000), ("target_train", 72000))


def task_plan(split, first_seed, episodes, horizon):
    return [
        {
            "episode_id": f"ep-{n:04d}",
            "seed": first_seed + n,
            "task": mini.task(split, first_seed + n),
            "main_opportunity": random.Random("checkpoint:" + str(first_seed + n)).randint(1, horizon),
            "q": 1 / horizon,
        }
        for n in range(episodes)
    ]


def register(args):
    bank_id, _, _ = open_bank(args.bank)
    reference = read(args.reference / "manifest.json")
    require(
        not reference["actor"]["do_sample"] and reference["actor"]["max_new_tokens"] == 256,
        "expected the frozen greedy actor and 256-token generation cap",
    )
    cohorts = [
        {"split": s, "seed": seed, "episodes": 16, "tasks": task_plan(s, seed, 16, 4)}
        for s, seed in STRATA
    ]
    plan = {
        "schema": 1,
        "study": "mini-natural-50-50-v1",
        "cohorts": cohorts,
        "population": "equal mixture of two fixed mini task templates; natural unassisted actions",
        "max_steps": 4,
        "max_new_tokens": 256,
        "temperature": 0,
        "replicates": 1,
        "seed": 7331,
        "actor_digest": digest(reference["actor"]),
        "bank_id": bank_id,
        "total_tokens": 16384,
        "total_tools": 64,
        "lambda_tokens": 0.01,
        "lambda_tool": 0.001,
        "lambda_time": 0.01,
        "stopping": "include all 32 prespecified episodes, no replacement or follow-up sampling",
        "statistics": "episodes are units; equal template weights; report uncertainty conditional on these templates",
    }
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "protocol.json", {**plan, "plan_id": digest(plan)})
    print(args.out / "protocol.json")


def check_run(plan, split, run):
    manifest, rows = load_run(run)
    spec = next(c for c in plan["cohorts"] if c["split"] == split)
    require(
        manifest["split"] == split and not manifest.get("cohort"),
        "scripted or wrong cohort in natural pilot",
    )
    require(
        manifest["seed"] == spec["seed"]
        and manifest["max_steps"] == plan["max_steps"]
        and digest(manifest["actor"]) == plan["actor_digest"],
        "incorrect seed, budget or actor",
    )
    expected = task_plan(split, spec["seed"], spec["episodes"], plan["max_steps"])
    require(
        expected == spec["tasks"] and len(rows) == len(expected),
        "task manifest differs from preregistration",
    )
    for row, designed in zip(rows, expected, strict=True):
        require(all(row[k] == designed[k] for k in designed), "episode differs from preregistration")
        selected_turn(run, manifest, row)
    return manifest, rows


def load_plan(path):
    plan = read(path)
    if "tasks" in plan and "cohorts" not in plan:
        pid = digest(plan)
        require(
            plan["protocol"] == protocol_digest()
            and plan["temperature"] == 0
            and plan["max_new_tokens"] == plan["actor"]["max_new_tokens"],
            "pilot actor or protocol drift",
        )
        groups = {}
        for row in plan["tasks"]:
            require(row["task_digest"] == digest(row["task"]), "preregistered task digest mismatch")
            groups.setdefault(row["stratum"], []).append(
                {k: v for k, v in row.items() if k not in ("stratum", "task_digest")}
            )
        require(set(groups) == {s for s, _ in STRATA}, "unexpected strata")
        plan.update(
            cohorts=[
                {"split": s, "seed": rows[0]["seed"], "episodes": len(rows), "tasks": rows}
                for s, rows in groups.items()
            ],
            actor_digest=digest(plan["actor"]),
            seed=plan["branch_seed"],
        )
        return plan, pid
    pid = plan.pop("plan_id")
    require(pid == digest(plan), "preregistration digest mismatch")
    return plan, pid


def script_exposed(args):
    """Pay for the original actor proposal, then execute a preregistered replacement at the anchor."""
    actor = FrozenQwen(args.model, 256, 0.0)
    parser = xml_parser()
    spec = mini.task("controlled_transfer", args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "run_id": uuid.uuid4().hex,
        "benchmark": "mini-data-v1",
        "cohort": "controlled_fixed_prefix",
        "split": "controlled_transfer",
        "seed": args.seed,
        "tasks": [spec],
        "actor": actor.info,
        "max_steps": 4,
        "memory_enabled": False,
        "teacher_enabled": False,
        "parser": type(parser).__name__,
        "split_plan": mini.TEMPLATES,
        "draft_origin": "scripted_POSITIVE_with_paid_unassisted_shadow",
        "fixture": "force first ASC and second POSITIVE regardless of the paid shadow proposal",
        "inference_scope": "engineering control only; exclude from natural frequency and success estimates",
    }
    save(args.out / "manifest.json", manifest)

    class InterventionFixture:
        def __init__(self):
            self.turn = 0

        def generate(self, messages, seed):
            self.turn += 1
            if self.turn > 2:
                return actor.generate(messages, seed)
            op = "ASC" if self.turn == 1 else "POSITIVE"
            response = (
                "<response><analysis></analysis><plan></plan><commands>"
                f'<keystrokes duration="0">{op}</keystrokes></commands></response>'
            )
            if self.turn == 1:
                return {
                    "text": response,
                    "status": "ok",
                    "origin": "scripted_control_prefix",
                    "seed": seed,
                    "seconds": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "prompt_digest": digest(actor.prompt(messages)),
                }
            original = actor.generate(messages, seed)
            natural, error = parse(original["text"], parser)
            return {
                **original,
                "text": response,
                "origin": "scripted_exposure_override",
                "shadow_actor_response": original,
                "shadow_actor_action": natural,
                "shadow_parse_error": error,
                "executed_draft_source": "preregistered scripted POSITIVE; shadow response never executed",
            }

    item = collect_episode(
        args.out, InterventionFixture(), parser, spec, "ep-0000", args.seed, 4, fixed_opportunity=2
    )
    item.update(
        run_id=manifest["run_id"], split=manifest["split"], draft_origin=manifest["draft_origin"]
    )
    (args.out / "index.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
    print(args.out)


def descending_source(args):
    """Separate source discovery run; never participates in the registered pilot."""
    actor = FrozenQwen(args.model, 256, 0.0)
    parser = xml_parser()
    tasks = []
    for n in range(6):
        spec = mini.task("source", 91000 + n)
        spec["family"] = "descending"
        spec["instruction"] = (
            "Order the entire sequence from greatest to least, retaining repeated integers, and submit the result."
        )
        tasks.append(spec)
    manifest = {
        "schema": 1,
        "run_id": uuid.uuid4().hex,
        "benchmark": "mini-data-v1",
        "cohort": "new_source_discovery",
        "split": "source",
        "seed": 91000,
        "tasks": tasks,
        "actor": actor.info,
        "max_steps": 4,
        "memory_enabled": False,
        "teacher_enabled": False,
        "parser": type(parser).__name__,
        "selection": "first independently graded failure in episode order; teacher sees one source trace only",
    }
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "manifest.json", manifest)
    with (args.out / "index.jsonl").open("x", encoding="utf-8") as stream:
        for n, task in enumerate(tasks):
            row = collect_episode(args.out, actor, parser, task, f"ep-{n:04d}", 91000 + n, 4)
            row.update(run_id=manifest["run_id"], split="source")
            stream.write(json.dumps(row) + "\n")
            stream.flush()


def shadow_cost(run, manifest, rows, cards, matcher=match):
    measurements = []
    for row in rows:
        turn = selected_turn(run, manifest, row)
        base_seconds = sum(
            read(run / n["checkpoint_id"] / "generation.json")["seconds"]
            for n in row["steps"]
            if (run / n["checkpoint_id"] / "generation.json").is_file()
        )
        missing_generation = any(
            not (run / n["checkpoint_id"] / "generation.json").is_file() for n in row["steps"]
        )
        result = {
            "episode_id": row["episode_id"],
            "run_id": manifest["run_id"],
            "bare_actor_seconds": base_seconds,
            "bare_actor_calls": len(row["steps"]),
            "bare_input_tokens": row["input_tokens"],
            "bare_output_tokens": row["output_tokens"],
            "bare_tool_calls": row["tool_calls"],
            "shadow_additional_actor_calls": 0,
            "shadow_additional_tokens": 0,
            "shadow_additional_tool_calls": 0,
        }
        if missing_generation:
            for key in ("bare_actor_seconds", "bare_input_tokens", "bare_output_tokens"):
                result[key] = None
        if turn is None:
            result.update(status="never_reached", retrieval_calls=0, match_seconds=0.0)
        else:
            pre, draft, _ = restore(run, row, turn)
            start = time.perf_counter()
            decision = matcher(cards, row["task"], pre, draft)
            elapsed = time.perf_counter() - start
            result.update(status=decision["status"], retrieval_calls=1, match_seconds=elapsed)
        measurements.append(result)
    return measurements


def audit_cohort(args):
    plan, pid = load_plan(args.plan)
    manifest, rows = check_run(plan, args.split, args.run)
    bank_id, cards, _ = open_bank(args.bank)
    require(plan["bank_id"] == bank_id, "wrong bank version")
    args.out.mkdir(parents=True, exist_ok=False)
    shadow = shadow_cost(args.run, manifest, rows, cards)
    save(
        args.out / "shadow.json",
        {
            "plan_id": pid,
            "run_id": manifest["run_id"],
            "rows": shadow,
            "measurement": "offline execution of the same in-memory match function, one per reached scheduled point; "
            "archives already loaded, no actor or tool rerun; not end-to-end wall-clock latency",
        },
    )
    common = {
        "run": args.run,
        "seed": plan["seed"],
        "total_tokens": plan["total_tokens"],
        "total_tools": plan["total_tools"],
    }
    # This preserves deterministic tape A/A for every task. Free A/A is only
    # necessary if this stratum actually contains an applicable opportunity.
    free = any(r["status"] in {"avoided", "exposed", "unknown_exposure"} for r in shadow)
    audit_aa(SimpleNamespace(**common, out=args.out / "aa", free=free))
    pair(
        SimpleNamespace(
            **common,
            out=args.out / "pairs",
            bank=args.bank,
            aa=args.out / "aa/aa.json",
            replicates=plan["replicates"],
            lambda_time=plan["lambda_time"],
            lambda_tokens=plan["lambda_tokens"],
            lambda_tool=plan["lambda_tool"],
        )
    )


def main():
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("plan")
    r.add_argument("--reference", type=Path, required=True)
    r.add_argument("--bank", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("script-exposed")
    c.add_argument("--seed", type=int, default=83000)
    c.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-7B")
    c.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("collect-desc-source")
    d.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-7B")
    d.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("audit")
    a.add_argument("--split", choices=[x[0] for x in STRATA], required=True)
    for name in ("plan", "run", "bank", "out"):
        a.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    {
        "plan": register,
        "audit": audit_cohort,
        "script-exposed": script_exposed,
        "collect-desc-source": descending_source,
    }[args.mode](args)


if __name__ == "__main__":
    main()
