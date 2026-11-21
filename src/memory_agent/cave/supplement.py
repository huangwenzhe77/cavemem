"""Audit and export the preregistered training-only enrichment against its immutable parent."""

import argparse
import json
import math
import random
import tempfile
from collections import Counter
from pathlib import Path

from . import formal, formal_plan as fp, prospective, supplement_plan as registration
from .pilot_report import reconcile
from .records import digest, read, save
from .replay import require, restore


TARGET_RULE = "more-expensive-silent-pass/v1"
TARGETS = ("costly-regression", "costly-both-pass")
CELLS = ("regression", "both_pass", "rescue", "both_fail")


def jsonlines(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def validate_plan(root, parent):
    plan, pid = fp.load(root)
    base, base_id = fp.load(parent)
    data = parent / "dataset-v1"
    release = read(data / "dataset.json")
    inherited = set(base) - {"cohorts", "schema", "split_counts", "design"}
    require(
        all(plan[k] == base[k] for k in inherited)
        and plan["schema"] == "formal-training-supplement/v1"
        and plan["target_rule_version"] == TARGET_RULE
        and plan["parent_plan_id"] == base_id == release["plan_id"]
        and plan["parent_dataset_id"]
        == release["dataset_id"]
        == digest({k: v for k, v in release.items() if k != "dataset_id"})
        and read(root / "sealed-test.json")
        == read(parent / "sealed-test.json")
        == read(data / "sealed-test.json")
        and release["code"] == formal.pipeline_signature()
        and release["engine"] == plan["protocol"]
        and release["bank_id"] == plan["bank_id"]
        and release["test_manifest_digest"] == plan["test_manifest_digest"]
        and plan["registration_code"] == fp.code_digest("supplement_plan.py")
        and [c["split"] for c in plan["cohorts"]] == ["train"],
        "supplement, parent dataset, registration, or held-out seal mismatch",
    )
    for s in fp.RUN_SPLITS:
        # Validate the frozen parent release; no parent outcomes are passed to
        # selection, labels or exports.
        for filename, key in (
            (s, "samples"),
            (s + "-episodes", "ledger"),
            (s + "-evidence", "evidence"),
        ):
            require(
                digest(jsonlines(data / (filename + ".jsonl")))
                == release["splits"][s][key + "_digest"],
                "parent dataset file has changed",
            )
    prior = [r["task"]["input"] for c in base["cohorts"] for r in c["tasks"]]
    prior += [r["task"]["input"] for r in read(parent / "sealed-test.json")["tasks"]]
    priors = fp.tasks_for(plan, "train")
    require(
        plan["split_counts"] == {"train": len(priors), "validation": 0, "test": 0},
        "wrong split counts",
    )
    templates = {g[1]: (g[2], g[5]) for g in fp.GROUPS if g[0] == "train"}
    groups = Counter()
    for i, item in enumerate(priors):
        t = item["task"]
        group, shape = t["template_id"], item["input_stratum"]
        seed = item["seed"]
        require(
            group in templates
            and templates[group] == (t["family"], t["instruction"])
            and shape in registration.SHAPES
            and item["episode_id"] == f"ep-{i:04d}"
            and item["task_digest"] == digest(t)
            and item["input_digest"] == digest(t["input"])
            and item["multiset_digest"] == digest(sorted(t["input"]))
            and registration.make_input(seed, shape, t["family"]) == t["input"]
            and item["q"] == 1 / plan["max_steps"]
            and item["main_opportunity"]
            == random.Random("checkpoint:" + str(seed)).randint(1, plan["max_steps"])
            and not any(fp.near_duplicate(t["input"], x) for x in prior),
            "unexpected template, signed input, schedule, or near-duplicate task",
        )
        prior.append(t["input"])
        groups[group, shape] += 1
    require(
        groups
        == Counter(
            {
                (g, s): n // len(registration.SHAPES)
                for g, n in registration.GROUP_COUNTS.items()
                for s in registration.SHAPES
            }
        ),
        "registered template by input-type counts changed",
    )
    return plan, pid


def lineage(args):
    plan, pid = validate_plan(args.root, args.parent)
    mf, rows = fp.checked_run(plan, "train", args.root / "train")
    cards = prospective.verified_cards(
        plan, args.parent_bank, args.source_package, args.previous_plan
    )
    return plan, pid, mf, rows, cards


def audit_seal(root, plan, pid, run_id):
    audit_dir = root / "audit-train"
    files = {
        p.relative_to(audit_dir).as_posix(): digest(read(p))
        for p in sorted(audit_dir.rglob("*.json"))
        if p != audit_dir / "supplement-seal.json"
    }
    return {
        "schema": "train-only-fork-audit/v1",
        "plan_id": pid,
        "run_id": run_id,
        "parent_dataset_id": plan["parent_dataset_id"],
        "target_rule_version": TARGET_RULE,
        "fork_code": formal.pipeline_signature(),
        "supplement_code": fp.code_digest("supplement.py"),
        "blind_seal": digest(read(root / "train/blind-seal.json")),
        "files": files,
    }


def audit(args):
    plan, pid, mf, rows, cards = lineage(args)
    out = args.root / "audit-train"
    prospective.audit_run(
        out, args.root / "train", mf, rows, plan, pid, cards, fp.formal_route
    )
    save(out / "supplement-seal.json", audit_seal(args.root, plan, pid, mf["run_id"]))


def classify(arms, plan):
    s, r = (arms[k] for k in ("silent", "remind"))
    dtok = r["total_tokens"] - s["total_tokens"]
    dsec = r["accounted_seconds"] - s["accounted_seconds"]
    dtool = r["tool_calls"] - s["tool_calls"]
    price = (
        plan["lambda_tokens"] * dtok / 1000
        + plan["lambda_time"] * dsec / 60
        + plan["lambda_tool"] * dtool
    )
    require(
        all(math.isfinite(n) for n in (dtok, dsec, dtool, price)),
        "incomparable branch usage",
    )
    ys, yr = s["success"], r["success"]
    group = None
    if ys == 1 and yr in (0, 1) and dtok > 0 and dsec > 0 and price > 0:
        group = TARGETS[yr]
    return {
        "Y_silent": ys,
        "Y_remind": yr,
        "delta_total_tokens": dtok,
        "delta_accounted_seconds": dsec,
        "delta_tool_calls": dtool,
        "delta_priced_cost": price,
        "target_subset": group,
    }


def build(root, plan, pid, mf, rows, cards):
    run, out = root / "train", root / "audit-train"
    seal = audit_seal(root, plan, pid, mf["run_id"])
    require(
        read(out / "supplement-seal.json") == seal,
        "changed fork evidence or single-train seal",
    )
    pdir = out / "pairs"
    policy = read(pdir / "protocol.json")
    require(
        policy["split"] == "train"
        and policy["router_version"] == fp.ROUTER
        and policy["plan_id"] == pid
        and policy["protocol"] == plan["protocol"],
        "branch protocol mismatch",
    )
    decisions, _ = prospective.schedule(run, mf, rows, cards, fp.formal_route)
    require(decisions == read(pdir / "matches.json"), "prefix-only rematch disagrees")
    checks = read(out / "aa/aa.json")["checks"]
    raw_index = {p["pair_id"]: p for p in read(pdir / "pairs.json") if "pair_id" in p}
    outcomes = reconcile(root, plan, pid, {"split": "train"}, run_reader=fp.checked_run)
    samples, evidence, ledger = [], [], []
    for row, reg, d, aa, result in zip(
        rows, fp.tasks_for(plan, "train"), decisions, checks, outcomes, strict=True
    ):
        eid = row["episode_id"]
        require(
            all(v["episode_id"] == eid for v in (reg, d, aa, result)),
            "episode mismatch",
        )
        line = {
            "episode_id": eid,
            "run_id": mf["run_id"],
            "task_digest": reg["task_digest"],
            "template_id": row["task"]["template_id"],
            "input_stratum": reg["input_stratum"],
            "seed": row["seed"],
            "q": row["q"],
            "scheduled_opportunity": row["main_opportunity"],
            "reached": result["reached"],
            "decision": d["status"],
            "rho": d.get("rho"),
            "e": d.get("e"),
            "checkpoint_id": d.get("checkpoint_id"),
            "tape_aa": result["tape_aa"],
            "free_aa": result["free_aa"],
            "pair_id": None,
            "sample_id": None,
            "cell": None,
            "target_subset": None,
            "exclusion": list(result["unpaired_reasons"]),
        }
        for obs in result["pairs"]:
            raw = raw_index[obs["pair_id"]]
            line["pair_id"] = raw["pair_id"]
            targets = formal.check_observation(
                pdir, raw, obs, d, plan, policy, row["task"], aa, run, row
            )
            evidence_row = {
                "pair_id": raw["pair_id"],
                "sample_id": None,
                "episode_id": eid,
                "plan_id": pid,
                "bank_id": plan["bank_id"],
                "template_id": line["template_id"],
                "input_stratum": line["input_stratum"],
                "task_digest": line["task_digest"],
                "checkpoint": "train/" + d["checkpoint_id"],
                "decision_digest": digest(d),
                "pair_record_digest": digest(raw),
                "cell": obs["cell"],
                "status": raw["status"],
                "arms": {
                    a: {
                        "path": "audit-train/pairs/" + raw["arms"][a]["file"],
                        "digest": raw["arms"][a]["digest"],
                        "observed": obs["arms"][a],
                    }
                    for a in ("silent", "remind")
                },
            }
            if targets is None:
                line["exclusion"].append("incomplete_pair")
            else:
                pre, draft, _ = restore(run, row, d["turn"])
                require(
                    digest(pre) == d["pre_digest"]
                    and draft["a0_digest"] == d["a0_digest"],
                    "wrong cached checkpoint",
                )
                sid = digest(
                    [pid, "train", mf["run_id"], d["checkpoint_id"], raw["pair_id"]]
                )
                cls = classify(obs["arms"], plan)
                require(
                    math.isclose(
                        cls["Y_remind"] - cls["Y_silent"] - cls["delta_priced_cost"],
                        targets["D"],
                        abs_tol=1e-9,
                    ),
                    "target costs disagree with D",
                )
                samples.append(
                    {
                        "sample_id": sid,
                        "features": formal.gate_inputs(row, pre, draft, d, plan),
                        "targets": targets,
                    }
                )
                evidence_row.update(sample_id=sid, **cls)
                line.update(
                    sample_id=sid, cell=obs["cell"], target_subset=cls["target_subset"]
                )
            evidence.append(evidence_row)
        if not line["pair_id"] and not line["exclusion"]:
            line["exclusion"] = [d["status"]]
        ledger.append(line)
    require(
        {e["pair_id"] for e in evidence} == set(raw_index)
        and len(evidence) == len(raw_index),
        "pair denominator mismatch",
    )
    return samples, evidence, ledger, digest(seal)


def counts(rows):
    completed = [r for r in rows if r["sample_id"]]
    cells = Counter(r["cell"] for r in completed)
    targets = Counter(r["target_subset"] for r in completed)
    return {
        "episodes": len(rows),
        "reached": sum(r["reached"] for r in rows),
        "applicable": sum(r["rho"] == 1 for r in rows),
        "pairs_attempted": sum(bool(r["pair_id"]) for r in rows),
        "complete_pairs": len(completed),
        "cells": {k: cells[k] for k in CELLS},
        "target_subsets": {k: targets[k] for k in TARGETS},
        "other_complete_pairs": targets[None],
        "nonlabel_reasons": dict(
            Counter(
                reason for r in rows if not r["sample_id"] for reason in r["exclusion"]
            )
        ),
    }


def export(args):
    plan, pid, mf, rows, cards = lineage(args)
    require(not args.out.exists(), "versioned dataset already exists")
    samples, evidence, ledger, seal = build(args.root, plan, pid, mf, rows, cards)
    subset = {r["sample_id"]: r["target_subset"] for r in ledger if r["sample_id"]}
    require(len(subset) == len(samples), "sample identity collision")
    files = {
        "train": samples,
        "train-evidence": evidence,
        "train-episodes": ledger,
        **{g: [s for s in samples if subset[s["sample_id"]] == g] for g in TARGETS},
    }
    by_group = {
        g: {
            shape: counts(
                [
                    r
                    for r in ledger
                    if (r["template_id"], r["input_stratum"]) == (g, shape)
                ]
            )
            for shape in registration.SHAPES
        }
        for g in registration.GROUP_COUNTS
    }
    release = {
        "schema": "formal-training-supplement-dataset/v1",
        "plan_id": pid,
        "run_id": mf["run_id"],
        "parent_plan_id": plan["parent_plan_id"],
        "parent_dataset_id": plan["parent_dataset_id"],
        "test_manifest_digest": plan["test_manifest_digest"],
        "bank_id": plan["bank_id"],
        "audit_seal": seal,
        "target_rule_version": TARGET_RULE,
        "label_spec_digest": plan["label_spec_digest"],
        "summary": counts(ledger),
        "by_template_and_input_stratum": by_group,
        "files": {
            n + ".jsonl": {"rows": len(v), "digest": digest(v)}
            for n, v in files.items()
        },
        "scope": "Training-only enrichment. The two outcome-selected subsets are overlapping views of the full training file and disjoint from one another. Frequencies describe this cohort.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()
        for name, data in files.items():
            formal.write_rows(stage / (name + ".jsonl"), data)
        save(stage / "dataset.json", {**release, "dataset_id": digest(release)})
        stage.rename(args.out)
    print(args.out / "dataset.json")


def main():
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    e = sub.add_parser("export")
    e.add_argument("--out", required=True, type=Path)
    for cmd in (a, e):
        cmd.add_argument("--root", required=True, type=Path)
        cmd.add_argument("--parent", type=Path, default=fp.OUTPUT / "cave-formal-01")
        cmd.add_argument("--parent-bank", type=Path, default=fp.DEFAULT_BANK)
        cmd.add_argument("--source-package", type=Path, default=fp.DEFAULT_PACKAGE)
        cmd.add_argument(
            "--previous-plan",
            type=Path,
            default=fp.OUTPUT / "cave-pilot-01/preregistration.json",
        )
    args = cli.parse_args()
    (audit if args.command == "audit" else export)(args)


if __name__ == "__main__":
    main()
