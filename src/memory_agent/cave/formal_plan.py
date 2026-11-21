"""Card 08 registration and blind collection. The sealed split has no execution entry point."""

import argparse
import hashlib
import json
import random
import uuid
from collections import Counter
from pathlib import Path

from . import mini
from .__main__ import collect_episode
from .experiment import archive_digest
from .models import FrozenQwen, xml_parser
from .prospective import route
from .records import digest, read, save
from .replay import load_run, protocol_digest, require, selected_turn


OUTPUT = Path("outputs")
DEFAULT_BANK = OUTPUT / "cave-replay-01/release/bank/bank.json"
DEFAULT_PACKAGE = OUTPUT / "cave-source-expansion-01"
RUN_SPLITS = ("train", "validation")
ROUTER = "formal-full-list-template-alias/v1"
GROUPS = (
    ("train", "ledger-up", "ascending", 210000, 12, "The ledger needs its entries in nondecreasing order. Submit every entry, including all copies of equal values."),
    ("train", "ledger-down", "descending", 211000, 12, "The ledger needs its entries in nonincreasing order. Submit every entry, including all copies of equal values."),
    ("train", "archive-up", "ascending", 212000, 12, "Reorder the archive so that the smallest integer appears first and the largest last. File the entire sorted sequence without dropping repeated entries."),
    ("train", "archive-down", "descending", 213000, 12, "Reorder the archive so that the largest integer appears first and the smallest last. File the entire sorted sequence without dropping repeated entries."),
    ("validation", "readings-up", "ascending", 220000, 12, "Prepare the readings for submission: start with the lowest reading and finish with the highest. Every occurrence must appear in the submitted sequence."),
    ("validation", "readings-down", "descending", 221000, 12, "Prepare the readings for submission: start with the highest reading and finish with the lowest. Every occurrence must appear in the submitted sequence."),
    ("test", "queue-up", "ascending", 230000, 8, "Submit a queue of all the given integers. A number may follow another only if it is at least as large. Keep the original multiplicity of each number."),
    ("test", "queue-down", "descending", 231000, 8, "Submit a queue of all the given integers. A number may follow another only if it is at most as large. Keep the original multiplicity of each number."),
)
LABELS = {
    "version": "formal-paired-labels/v1", "D": "reward(remind)-reward(silent) minus observed incremental priced costs",
    "H": "int(silent_success == 1 and remind_success == 0)", "sensitivity_cost_multipliers": [0, 0.5, 1, 2],
    "m_label": 0.002, "policy_risk_delta": 0.05, "policy_mean_extra_tokens_budget": 256,
    "K": 1, "hard_label": "uncertain", "hard_loss_weight": 0,
    "scope": "Greedy single paired observation; no per-checkpoint uncertainty interval. Policy budgets are design assumptions.",
}


def code_digest(filename):
    data = Path(__file__).with_name(filename).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def near_duplicate(a, b):
    return len(a) == len(b) and len(a) - sum((Counter(a) & Counter(b)).values()) <= 1


def templates_and_tasks(previous):
    seen = list(previous)
    cohorts = {s: [] for s in (*RUN_SPLITS, "test")}
    for split, group, family, start, count, instruction in GROUPS:
        rows = cohorts[split]
        for seed in range(start, start + count):
            values = mini.task("source", seed)["input"]
            require(not any(near_duplicate(values, old) for old in seen), "duplicate or near-duplicate input")
            seen.append(values)
            task = {"template_id": group, "family": family, "instruction": instruction, "input": values}
            rows.append({
                "episode_id": f"ep-{len(rows):04d}", "task": task, "seed": seed,
                "task_digest": digest(task), "input_digest": digest(values),
                "multiset_digest": digest(sorted(values)), "q": .25,
                "main_opportunity": random.Random("checkpoint:" + str(seed)).randint(1, 4),
            })
    return cohorts


def formal_route(cards, task, pre, draft):
    group = next((g for g in GROUPS if g[1] == task["template_id"]), None)
    require(group and (task["family"], task["instruction"]) == (group[2], group[5]), "unknown template or changed semantic alias")
    # Alias only the already-reviewed task semantics. Keep real observable state,
    # historical messages and complete draft untouched for the predicate.
    semantic = {**task, "instruction": mini.TEMPLATES["target_train" if group[2] == "descending" else "controlled_transfer"][1]}
    return route(cards, semantic, pre, draft)


def freeze(root, pilot1, pilot2, source1, package, bank):
    first, second = read(pilot1), read(pilot2)
    source = read(source1)
    other = read(package / "source/manifest.json")
    parent = read(bank)
    require(
        digest(first) == second["previous_plan_digest"] and len(first["tasks"]) == len(second["tasks"]) == 32
        and first["bank_id"] == second["parent_bank_id"] == parent["bank_id"]
        and parent["source_manifest_digest"] == digest(source)
        and source["split"] == other["split"] == "source"
        and second["descending_memory_id"] == read(package / "development-v2/card.json")["memory_id"]
        and second["protocol"] == protocol_digest() and second["actor"]["do_sample"] is False
        and second["actor"]["max_new_tokens"] == 256 and second["actor"]["temperature"] == 0,
        "pilot, source, engine or actor lineage changed",
    )
    prior = [t["task"]["input"] for p in (first, second) for t in p["tasks"]]
    prior.extend(t["input"] for m in (source, other) for t in m["tasks"])
    cohorts = templates_and_tasks(prior)
    sealed = {"schema": "sealed-formal-test/v1", "tasks": cohorts.pop("test"), "execution": "sealed; registration only"}
    plan = {
        "schema": "formal-mini-08/v1", "protocol": protocol_digest(), "collector": code_digest("formal_plan.py"),
        "base_collector": code_digest("__main__.py"), "previous_plan_digest": digest(first), "pilot_02_digest": digest(second),
        "source_manifest_digests": [digest(source), digest(other)], "excluded_input_digests": [digest(x) for x in prior],
        "dedup": "Reject equal-length multisets differing by at most one substitution, against both sources, both pilots and every formal task.",
        "router_version": ROUTER, "templates_digest": digest(GROUPS), "label_spec": LABELS, "label_spec_digest": digest(LABELS),
        "actor": second["actor"], "bank_id": second["bank_id"], "parent_bank_id": second["parent_bank_id"],
        "descending_memory_id": second["descending_memory_id"], "max_steps": 4, "total_tokens": 16384,
        "total_tools": 64, "replicates": 1, "seed": 82008, "lambda_tokens": .01, "lambda_time": .01, "lambda_tool": .001,
        "design": "Independent task-wording groups per split, same two mechanisms. Natural greedy actor; q=1/4; one scheduled opportunity, no refill. Both complete blind ledgers before matching.",
        "test_manifest_digest": digest(sealed), "split_counts": {"train": 48, "validation": 24, "test": 16},
        "cohorts": [{"split": s, "tasks": rows} for s, rows in cohorts.items()],
    }
    root.mkdir(parents=True, exist_ok=False)
    save(root / "sealed-test.json", sealed)
    save(root / "preregistration.json", {**plan, "plan_id": digest(plan)})
    print("plan", digest(plan), "sealed test", digest(sealed), flush=True)


def load(root):
    p = read(root / "preregistration.json")
    pid = p.pop("plan_id")
    require(pid == digest(p) and p["collector"] == code_digest("formal_plan.py")
            and p["base_collector"] == code_digest("__main__.py") and p["protocol"] == protocol_digest()
            and p["templates_digest"] == digest(GROUPS) and p["label_spec_digest"] == digest(LABELS)
            and p["label_spec"] == LABELS and p["router_version"] == ROUTER
            and digest(read(root / "sealed-test.json")) == p["test_manifest_digest"], "registration, test seal or implementation changed")
    return p, pid


def tasks_for(plan, split):
    require(split in RUN_SPLITS, "sealed split cannot be executed")
    return next(c["tasks"] for c in plan["cohorts"] if c["split"] == split)


def collect(root, split):
    plan, pid = load(root)
    tasks = tasks_for(plan, split)
    run = root / split
    require(not run.exists(), "collection already exists")
    actor = FrozenQwen(plan["actor"]["actor"], 256, 0)
    require(actor.info == plan["actor"], "frozen actor drift")
    parser = xml_parser()
    run.mkdir()
    rid = uuid.uuid4().hex
    mf = {"schema": 1, "run_id": rid, "benchmark": "mini-data-v1", "split": split, "formal_plan_id": pid,
          "tasks": [r["task"] for r in tasks], "actor": actor.info, "parser": type(parser).__name__, "max_steps": 4,
          "cohort": "natural", "memory_enabled": False, "teacher_enabled": False}
    save(run / "manifest.json", mf)
    with (run / "index.jsonl").open("x", encoding="utf-8") as f:
        for row in tasks:
            item = collect_episode(run, actor, parser, row["task"], row["episode_id"], row["seed"], 4)
            require(item["main_opportunity"] == row["main_opportunity"] and item["q"] == row["q"], "opportunity drift")
            item.update(run_id=rid, split=split)
            f.write(json.dumps(item) + "\n")
            f.flush()
            print(item["episode_id"], item["termination"], flush=True)
    _, rows = load_run(run)
    save(run / "blind-seal.json", {"plan_id": pid, "manifest_digest": digest(mf), "episodes": [archive_digest(run, r) for r in rows]})


def checked_run(plan, split, run):
    priors = tasks_for(plan, split)
    mf, rows = load_run(run)
    pid = digest(plan)
    require(mf["formal_plan_id"] == pid and mf["actor"] == plan["actor"] and mf["split"] == split
            and mf["cohort"] == "natural" and mf["max_steps"] == plan["max_steps"] and len(rows) == len(priors),
            "wrong formal baseline")
    for r, t in zip(rows, priors, strict=True):
        require(all(r[k] == t[k] for k in ("episode_id", "seed", "main_opportunity", "q", "task"))
                and digest(r["task"]) == t["task_digest"], "not the preregistered episode")
        selected_turn(run, mf, r)
    require(read(run / "blind-seal.json") == {"plan_id": pid, "manifest_digest": digest(mf),
            "episodes": [archive_digest(run, r) for r in rows]}, "blind archive was not completed or has changed")
    return mf, rows


def main():
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--pilot-01", type=Path, default=OUTPUT / "cave-pilot-01/preregistration.json")
    p.add_argument("--pilot-02", type=Path, default=OUTPUT / "cave-pilot-02/preregistration.json")
    p.add_argument("--source-01", type=Path, default=OUTPUT / "cave-acceptance/source/manifest.json")
    p.add_argument("--source-package", type=Path, default=DEFAULT_PACKAGE)
    p.add_argument("--parent-bank", type=Path, default=DEFAULT_BANK)
    c = sub.add_parser("collect")
    c.add_argument("--split", choices=RUN_SPLITS, required=True)
    for cmd in (p, c):
        cmd.add_argument("--root", type=Path, required=True)
    args = cli.parse_args()
    if args.command == "plan":
        freeze(args.root, args.pilot_01, args.pilot_02, args.source_01, args.source_package, args.parent_bank)
    else:
        collect(args.root, args.split)


if __name__ == "__main__":
    main()
