"""Run and export card 08 matched forks after blind collection is complete."""

import argparse
import math
import random
import tempfile
from collections import Counter
from pathlib import Path

from . import formal_plan as fp
from . import prospective
from .experiment import archive_digest
from .pilot_report import reconcile
from .records import digest, encode, read, save, verify
from .replay import require, restore


def lineage(args):
    plan, pid = fp.load(args.root)
    cards = prospective.verified_cards(
        plan, args.parent_bank, args.source_package, args.previous_plan
    )
    runs = {s: fp.checked_run(plan, s, args.root / s) for s in fp.RUN_SPLITS}
    require(
        len({m["run_id"] for m, rows in runs.values()}) == len(runs),
        "shared baseline run IDs",
    )
    return plan, pid, cards, runs


def pipeline_signature():
    return digest(
        {
            n: fp.code_digest(n)
            for n in (
                "formal.py",
                "prospective.py",
                "development_card.py",
                "pilot.py",
                "pilot_report.py",
                "records.py",
            )
        }
    )


def audit(args):
    plan, pid, cards, runs = lineage(args)
    mf, rows = runs[args.split]
    run = args.root / args.split
    out = args.root / ("audit-" + args.split)
    prospective.audit_run(out, run, mf, rows, plan, pid, cards, fp.formal_route)
    # Seal all decisions, both A/A traces and both treatment arms. The plan ID
    # separately covers the original actor, budgets, alias table and price rule.
    refs = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.json"))
    save(
        out / "formal-seal.json",
        {
            "plan_id": pid,
            "code": pipeline_signature(),
            "split": args.split,
            "run_id": mf["run_id"],
            "blind_seals": {
                s: digest(read(args.root / s / "blind-seal.json")) for s in runs
            },
            "files": {n: digest(read(out / n)) for n in refs},
        },
    )


def check_audit(root, split, pid, run_id):
    out = root / ("audit-" + split)
    seal = read(out / "formal-seal.json")
    refs = sorted(
        p.relative_to(out).as_posix()
        for p in out.rglob("*.json")
        if p != out / "formal-seal.json"
    )
    require(
        seal
        == {
            "plan_id": pid,
            "code": pipeline_signature(),
            "split": split,
            "run_id": run_id,
            "blind_seals": {
                s: digest(read(root / s / "blind-seal.json")) for s in fp.RUN_SPLITS
            },
            "files": {n: digest(read(out / n)) for n in refs},
        },
        "changed formal fork or archived evidence",
    )
    return digest(seal)


def label(obs, plan):
    """K=1 empirical targets, using raw token, tool and timing differences."""
    s, r = (obs["arms"][a] for a in ("silent", "remind"))
    require(
        s["success"] in (0, 1) and r["success"] in (0, 1),
        "cannot label a missing outcome",
    )
    dy = r["success"] - s["success"]
    penalty = (
        plan["lambda_tokens"] * (r["total_tokens"] - s["total_tokens"]) / 1000
        + plan["lambda_tool"] * (r["tool_calls"] - s["tool_calls"])
        + plan["lambda_time"] * (r["accounted_seconds"] - s["accounted_seconds"]) / 60
    )
    measured = dy - penalty
    require(
        math.isclose(measured, obs["delta"]["utility"], abs_tol=1e-9),
        "net utility does not reconcile",
    )
    return {
        "D": measured,
        "H": int(s["success"] == 1 and r["success"] == 0),
        "K": 1,
        "hard_label": "uncertain",
        "hard_loss_weight": 0,
        "continuous_loss_weight": 1,
        "D_cost_sensitivity": {
            str(w): dy - w * penalty
            for w in plan["label_spec"]["sensitivity_cost_multipliers"]
        },
        "hard_label_reason": "single greedy coupling; no independent checkpoint-level interval",
    }


def gate_inputs(row, pre, draft, decision, plan):
    # Explicit whitelist. Only fields observable before executing the first
    # branch are exposed to the gate. No verifier receipts or future trajectory.
    return {
        "instruction": row["task"]["instruction"],
        "history_before_proposal": pre["messages"],
        "working_state": pre["state"],
        "proposal": draft["a0"],
        "parse_warning": draft["warning"],
        "memory": {"id": decision["memory_id"], "reminder": decision["reminder"]},
        "evidence": {
            "source_verified": decision["v"],
            "applicable": decision["rho"],
            "exposed": decision["e"],
        },
        "budget_after_proposal": {
            "remaining_actor_calls": plan["max_steps"] - pre["step"],
            "remaining_execution_turns": pre["remaining_steps"],
            "remaining_tokens": plan["total_tokens"]
            - draft["used_input_tokens"]
            - draft["used_output_tokens"],
            "remaining_tools": plan["total_tools"] - pre["used_tool_calls"],
        },
    }


def check_observation(pdir, raw, obs, decision, plan, policy, spec, aa, run, row):
    expected = digest([digest(policy), decision, 0])
    flags = ["silent", "remind"]
    random.Random("order:" + expected).shuffle(flags)
    seed = random.Random(
        f"coupling:{plan['seed']}:{policy['split']}:{row['episode_id']}:0"
    ).randrange(2**31)
    require(
        raw["pair_id"] == expected
        and raw["order"] == flags
        and raw["seed"] == seed
        and raw["replicate"] == 0
        and aa["tape_status"] == "pass"
        and aa["free_status"] == "agree"
        and aa["free_anchor"] == decision["turn"]
        and aa["archive_digest"] == archive_digest(run, row),
        "fork identity or exact-anchor A/A mismatch",
    )
    prev = None
    receipts = set()
    pre, draft, _ = restore(run, row, decision["turn"])
    for side in ("silent", "remind"):
        ref = raw["arms"][side]
        data = read(pdir / ref["file"])
        u = data["usage"]
        require(
            data["seed"] == seed
            and data["limits"]
            == {
                "actor_calls": plan["max_steps"],
                "turns": plan["max_steps"],
                "total_tokens": plan["total_tokens"],
                "tool_calls": plan["total_tools"],
            }
            and u["actor_calls"] <= plan["max_steps"]
            and u["tool_calls"] <= plan["total_tools"]
            and u["input_tokens"] + u["output_tokens"] <= plan["total_tokens"]
            and data["revision_calls"] == int(side == "remind"),
            "branch resource or revision mismatch",
        )
        if prev is not None:
            require(
                prev == data["common_prefix"], "arms charged different original drafts"
            )
        prev = data["common_prefix"]
        if side == "silent" and data["events"]:
            require(
                data["events"][0]["proposed"] == draft["a0"],
                "silent did not use the cached proposal",
            )
        if side == "remind":
            request = data["requests"][0]
            expected_text = (
                pre["messages"][-1]["content"]
                + "\n\nAdvisory note for this proposal only: "
                + decision["reminder"]
            )
            require(
                request["stage"] == "revision"
                and request["messages"][:-1] == pre["messages"][:-1]
                and request["messages"][-1]["content"] == expected_text,
                "reminder was not applied to the original prefix",
            )
        if ref["Y"] is not None:
            receipt = data["score"].get("request_id")
            require(
                receipt
                and receipt not in receipts
                and data["score"]["status"] == ("pass" if ref["Y"] else "fail")
                and verify(spec, data["state"])["reward"] == ref["Y"],
                "independent terminal regrade failed",
            )
            receipts.add(receipt)
    if obs["cell"] is None:
        return None
    targets = label(obs, plan)
    require(
        targets["H"] == raw["regression"]
        and math.isclose(raw["difference"], targets["D"], abs_tol=1e-9),
        "raw D/H contradicts independent calculation",
    )
    return targets


def build_split(root, split, plan, pid, cards, manifest, rows):
    run = root / split
    audit_dir = root / ("audit-" + split)
    pdir = audit_dir / "pairs"
    seal = check_audit(root, split, pid, manifest["run_id"])
    decisions, _ = prospective.schedule(run, manifest, rows, cards, fp.formal_route)
    require(
        decisions == read(pdir / "matches.json"),
        "decision differs from prefix-only rematch",
    )
    policy = read(pdir / "protocol.json")
    require(
        policy["split"] == split
        and policy["run_id"] == manifest["run_id"]
        and policy["plan_id"] == pid
        and policy["router_version"] == fp.ROUTER
        and policy["protocol"] == plan["protocol"],
        "wrong branch protocol",
    )
    records = read(pdir / "pairs.json")
    raw_pairs = {r["pair_id"]: r for r in records if "pair_id" in r}
    checks = read(audit_dir / "aa/aa.json")["checks"]
    reconciled = reconcile(root, plan, pid, {"split": split}, run_reader=fp.checked_run)
    examples, ledger, evidence = [], [], []
    for row, result, decision, aa in zip(
        rows, reconciled, decisions, checks, strict=True
    ):
        eid = row["episode_id"]
        require(
            row["episode_id"]
            == result["episode_id"]
            == decision["episode_id"]
            == aa["episode_id"],
            "episode mismatch",
        )
        checkpoint = decision.get("checkpoint_id")
        entry = {
            "split": split,
            "template_id": row["task"]["template_id"],
            "task_digest": digest(row["task"]),
            "run_id": manifest["run_id"],
            "episode_id": eid,
            "seed": row["seed"],
            "q": row["q"],
            "scheduled_opportunity": row["main_opportunity"],
            "checkpoint_id": checkpoint,
            "decision": decision["status"],
            "tape_aa": result["tape_aa"],
            "free_aa": result["free_aa"],
            "exclusion": result["unpaired_reasons"],
            "pair_ids": [],
            "sample_id": None,
        }
        for obs in result["pairs"]:
            raw = raw_pairs[obs["pair_id"]]
            entry["pair_ids"].append(raw["pair_id"])
            targets = check_observation(
                pdir, raw, obs, decision, plan, policy, row["task"], aa, run, row
            )
            if targets is None:
                entry["exclusion"].append("incomplete_pair")
                continue
            pre, draft, _ = restore(run, row, decision["turn"])
            require(
                digest(pre) == decision["pre_digest"]
                and draft["a0_digest"] == decision["a0_digest"],
                "checkpoint mismatch",
            )
            sid = digest([pid, split, manifest["run_id"], checkpoint, raw["pair_id"]])
            examples.append(
                {
                    "sample_id": sid,
                    "features": gate_inputs(row, pre, draft, decision, plan),
                    "targets": targets,
                }
            )
            evidence.append(
                {
                    "sample_id": sid,
                    "plan_id": pid,
                    "bank_id": plan["bank_id"],
                    "pair_id": raw["pair_id"],
                    "task_digest": entry["task_digest"],
                    "checkpoint": f"{split}/{checkpoint}",
                    "decision_digest": digest(decision),
                    "pair_record_digest": digest(raw),
                    "cell": obs["cell"],
                    "arms": {
                        a: {
                            "path": f"audit-{split}/pairs/{raw['arms'][a]['file']}",
                            "digest": raw["arms"][a]["digest"],
                            "observed": obs["arms"][a],
                        }
                        for a in ("silent", "remind")
                    },
                }
            )
            entry["sample_id"] = sid
        if not entry["pair_ids"] and not entry["exclusion"]:
            entry["exclusion"] = [decision["status"]]
        ledger.append(entry)
    return examples, evidence, ledger, seal


def write_rows(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(encode(row) + "\n")


def export(args):
    plan, pid, cards, runs = lineage(args)
    require(not args.out.exists(), "versioned export already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        dest = Path(tmp) / "export"
        dest.mkdir()
        info = {}
        for split, (mf, rows) in runs.items():
            examples, evidence, ledger, seal = build_split(
                args.root, split, plan, pid, cards, mf, rows
            )
            for name, obj in (
                (split, examples),
                (split + "-evidence", evidence),
                (split + "-episodes", ledger),
            ):
                write_rows(dest / (name + ".jsonl"), obj)
            info[split] = {
                "episodes": len(ledger),
                "samples": len(examples),
                "samples_digest": digest(examples),
                "evidence_digest": digest(evidence),
                "ledger_digest": digest(ledger),
                "audit_seal": seal,
                "cells": dict(Counter(e["cell"] for e in evidence)),
            }
        save(dest / "sealed-test.json", read(args.root / "sealed-test.json"))
        release = {
            "schema": "formal-paired-dataset/v1",
            "plan_id": pid,
            "engine": plan["protocol"],
            "code": pipeline_signature(),
            "bank_id": plan["bank_id"],
            "test_manifest_digest": plan["test_manifest_digest"],
            "label_spec": plan["label_spec"],
            "label_spec_digest": plan["label_spec_digest"],
            "splits": info,
            "population": "One scheduled checkpoint per registered episode. Only two independently scored arms yield continuous supervision.",
        }
        save(dest / "dataset.json", {**release, "dataset_id": digest(release)})
        dest.rename(args.out)
    print(args.out / "dataset.json")


def main():
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--split", required=True, choices=fp.RUN_SPLITS)
    e = sub.add_parser("export")
    e.add_argument("--out", required=True, type=Path)
    for cmd in (a, e):
        cmd.add_argument("--root", required=True, type=Path)
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
