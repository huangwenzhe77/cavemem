"""Prospective validation of frozen card-09 policies on fresh sorting tasks."""

import argparse
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import formal, formal_plan as fp, prospective
from .pilot_report import reconcile
from .records import digest, read, save
from .replay import require, restore
from .supplement_plan import SHAPES, make_input


TEMPLATES = {
    "batch-down": ("descending", "Submit the entire batch in nonincreasing order. Preserve every occurrence of each integer."),
    "batch-up": ("ascending", "Submit the entire batch in nondecreasing order. Preserve every occurrence of each integer."),
    "list-down": ("descending", "Arrange the full list from greatest to least and submit it, including repeated numbers."),
    "list-up": ("ascending", "Arrange the full list from least to greatest and submit it, including repeated numbers."),
}


def register(root, parent):
    original, parent_id = fp.load(parent)
    sealed = read(parent / "sealed-test.json")
    seen = [t["task"]["input"] for t in sealed["tasks"]]
    for file in sorted(parent.parent.glob("cave-*/**/manifest.json")):
        mf = read(file)
        if mf.get("benchmark") == "mini-data-v1":
            seen.extend(t["input"] for t in mf["tasks"])
    seen = [list(t) for t in sorted({tuple(sorted(v)) for v in seen})]
    historical_digest = digest(seen)
    groups = []
    for i, (name, (family, instruction)) in enumerate(TEMPLATES.items()):
        seed = 610000 + 10000 * i
        for n in range(16):
            shape = SHAPES[n % len(SHAPES)]
            while True:
                seed += 1
                values = make_input(seed, shape, family)
                if values is not None and not any(fp.near_duplicate(t, values) for t in seen):
                    break
            seen.append(values)
            task = {"family": family, "instruction": instruction, "input": values, "template_id": name}
            groups.append({"episode_id": f"ep-{len(groups):04d}", "task": task, "seed": seed,
                           "task_digest": digest(task), "input_digest": digest(values),
                           "multiset_digest": digest(sorted(values)), "input_stratum": shape, "q": .25,
                           "main_opportunity": random.Random("checkpoint:" + str(seed)).randint(1, 4)})
    plan = {**original, "schema": "frozen-gate-validation/v1", "parent_plan_id": parent_id,
            "created_utc": datetime.now(timezone.utc).isoformat(), "past_inputs_digest": historical_digest,
            "validation_code": fp.code_digest("gate_validation.py"), "validation_templates": TEMPLATES,
            "split_counts": {"train": 0, "validation": len(groups), "test": 0},
            "cohorts": [{"split": "validation", "tasks": groups}],
            "design": "Four unseen wordings with equal counts; same two mechanisms. Four prespecified input "
                      "shapes per wording, uniform opportunity 1..4, no refill. Frozen training-only model. "
                      "Report silent/remind/ready-submit/learned/ablation before any threshold changes; "
                      "all episodes and observed pairs separately. No evaluation on the sealed test manifest."}
    root.mkdir(parents=True, exist_ok=False)
    save(root/"sealed-test.json", sealed)
    save(root/"preregistration.json", {**plan, "plan_id": digest(plan)})
    print(digest(plan), len(groups))


def route(cards, task, pre, draft):
    family, instruction = TEMPLATES[task["template_id"]]
    require((task["family"], task["instruction"]) == (family, instruction), "unregistered task wording")
    name = "controlled_transfer" if family == "ascending" else "target_train"
    semantic = {**task, "instruction": fp.mini.TEMPLATES[name][1]}
    return prospective.route(cards, semantic, pre, draft)


def evaluate_arms(root, model_file):
    plan, pid = fp.load(root)
    require(plan["validation_code"] == fp.code_digest("gate_validation.py"), "validation code changed")
    model_digest = digest(read(model_file))
    run = root/"validation"
    mf, rows = fp.checked_run(plan, "validation", run)
    cards = prospective.verified_cards(plan, fp.DEFAULT_BANK, fp.DEFAULT_PACKAGE,
                                       fp.OUTPUT/"cave-pilot-01/preregistration.json")
    audit_dir = root/"audit-validation"
    require(not audit_dir.exists(), "validation audit already exists")
    save(root/"model-before-pairing.json", {"plan_id": pid, "model_digest": model_digest,
         "blind_digest": digest(read(run/"blind-seal.json")), "utc": datetime.now(timezone.utc).isoformat()})
    prospective.audit_run(audit_dir, run, mf, rows, plan, pid, cards, route)
    pdir = audit_dir/"pairs"
    policy = read(pdir/"protocol.json")
    decisions, _ = prospective.schedule(run, mf, rows, cards, route)
    require(decisions == read(pdir/"matches.json"), "cached decisions drifted")
    checks = read(audit_dir/"aa/aa.json")["checks"]
    pairs = {r["pair_id"]: r for r in read(pdir/"pairs.json") if "pair_id" in r}
    rec = reconcile(root, plan, pid, {"split":"validation"}, run_reader=fp.checked_run)
    examples, evidence, ledger = [], [], []
    for row, d, a, result in zip(rows, decisions, checks, rec, strict=True):
        eid = row["episode_id"]
        require(eid == d["episode_id"] == a["episode_id"] == result["episode_id"], "misaligned task")
        entry = {"split": "validation", "template_id": row["task"]["template_id"], "task_digest": digest(row["task"]),
                 "run_id": mf["run_id"], "episode_id": eid, "seed": row["seed"], "q": row["q"],
                 "scheduled_opportunity": row["main_opportunity"], "checkpoint_id": d.get("checkpoint_id"),
                 "decision": d["status"], "tape_aa": result["tape_aa"], "free_aa": result["free_aa"],
                 "exclusion": result["unpaired_reasons"], "pair_ids": [], "sample_id": None}
        for obs in result["pairs"]:
            raw = pairs[obs["pair_id"]]
            entry["pair_ids"].append(raw["pair_id"])
            labels = formal.check_observation(pdir, raw, obs, d, plan, policy, row["task"], a, run, row)
            if labels is None:
                entry["exclusion"].append("incomplete_pair")
                continue
            pre, draft, _ = restore(run, row, d["turn"])
            sid = digest([pid, "validation", mf["run_id"], d["checkpoint_id"], raw["pair_id"]])
            examples.append({"sample_id": sid, "features": formal.gate_inputs(row, pre, draft, d, plan), "targets": labels})
            entry["sample_id"] = sid
            evidence.append({"sample_id": sid, "plan_id": pid, "bank_id": plan["bank_id"], "pair_id": raw["pair_id"],
                             "task_digest": entry["task_digest"], "checkpoint": "validation/"+d["checkpoint_id"],
                             "decision_digest": digest(d), "pair_record_digest": digest(raw), "cell": obs["cell"],
                             "arms": {k: {"path": "audit-validation/pairs/"+raw["arms"][k]["file"],
                                          "digest": raw["arms"][k]["digest"], "observed": obs["arms"][k]}
                                      for k in ("silent","remind")}})
        if not entry["pair_ids"] and not entry["exclusion"]:
            entry["exclusion"] = [d["status"]]
        ledger.append(entry)
    out = root/"dataset-v1"
    out.mkdir(exist_ok=False)
    for name, data in (("validation",examples),("validation-evidence",evidence),("validation-episodes",ledger)):
        formal.write_rows(out/(name+".jsonl"), data)
    info = {"episodes": len(ledger), "samples": len(examples), "samples_digest": digest(examples),
            "evidence_digest": digest(evidence), "ledger_digest": digest(ledger), "cells":dict(Counter(e["cell"] for e in evidence))}
    release = {"schema": "frozen-gate-validation/v1","plan_id":pid,"bank_id":plan["bank_id"],
               "test_manifest_digest":plan["test_manifest_digest"],"model_before_pairing":model_digest,
               "splits":{"validation":info}}
    save(out/"dataset.json",{**release,"dataset_id":digest(release)})
    print(info)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--parent",type=Path,default=fp.OUTPUT/"cave-formal-01")
    a = sub.add_parser("audit")
    a.add_argument("--model",type=Path,required=True)
    for item in (p,a):
        item.add_argument("--root",type=Path,required=True)
    opts = cli.parse_args()
    if opts.mode == "plan":
        register(opts.root, opts.parent)
    else:
        evaluate_arms(opts.root, opts.model)
