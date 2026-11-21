"""Fit and serve the card-09 gate using only frozen training exports."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from . import formal_plan as fp
from .records import digest, read, save
from .replay import require


BASE = ("descending", "already_ordered", "contains_nonpositive", "last_revision_slot")
ACTION = (
    "ready_to_submit",
    "contains_submit",
    "filters_records",
    "exposure_known",
    "exposed",
)
MIN_LEAF_TASK_WEIGHT = 4
DEFAULT_THRESHOLD = 0.002


def lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def signed(path, key):
    doc = read(path)
    require(
        doc[key] == digest({k: v for k, v in doc.items() if k != key}),
        "invalid signature: " + path.name,
    )
    return doc


def ready(features, cards):
    """Static interpretation of the fixed mini DSL; no verifier, model, or environment call."""
    direction = cards[features["memory"]["id"]]
    state = features["working_state"]
    working = list(state["working"])
    submitted = None
    for command in features["proposal"]["commands"]:
        op = command["keystrokes"].strip()
        if op == "ASC":
            working.sort()
        elif op == "DESC":
            working.sort(reverse=True)
        elif op == "POSITIVE":
            working = [n for n in working if n > 0]
        elif op == "UNIQUE":
            working = list(dict.fromkeys(working))
        elif op == "SUBMIT":
            submitted = list(working)
        # SHOW and rejected operations leave mini state unchanged.
    return submitted is not None and submitted == sorted(
        state["input"], reverse=direction == "descending"
    )


def features_for(x, cards, variant):
    require(variant in ("full", "no_a0"), "unknown feature set")
    desc = cards[x["memory"]["id"]] == "descending"
    state = x["working_state"]
    base = {
        "descending": desc,
        "already_ordered": state["working"] == sorted(state["input"], reverse=desc),
        "contains_nonpositive": any(n <= 0 for n in state["working"]),
        "last_revision_slot": x["budget_after_proposal"]["remaining_actor_calls"] <= 1,
    }
    if variant == "full":
        ops = [c["keystrokes"].strip() for c in x["proposal"]["commands"]]
        exposure = x["evidence"]["exposed"]
        base.update(
            ready_to_submit=ready(x, cards),
            contains_submit="SUBMIT" in ops,
            filters_records=bool({"POSITIVE", "UNIQUE"}.intersection(ops)),
            exposure_known=exposure is not None,
            exposed=exposure == 1,
        )
    return base


def stump(x, y, weights, columns):
    def leaf(indices):
        weight = sum(weights[i] for i in indices)
        avg = sum(weights[i] * y[i] for i in indices) / weight
        return weight, avg, sum(weights[i] * (y[i] - avg) ** 2 for i in indices)

    idx = list(range(len(y)))
    _, constant, loss = leaf(idx)
    best = {"split": None, "means": [constant]}
    for name in columns:
        left = [i for i in idx if not x[i][name]]
        right = [i for i in idx if x[i][name]]
        if not left or not right:
            continue
        n0, m0, l0 = leaf(left)
        n1, m1, l1 = leaf(right)
        if min(n0, n1) >= MIN_LEAF_TASK_WEIGHT and l0 + l1 < loss - 1e-12:
            best, loss = {"split": name, "means": [m0, m1]}, l0 + l1
    return best


def forecast(head, row):
    return head["means"][int(bool(row[head["split"]])) if head["split"] else 0]


def predict(model, features, variant="full"):
    row = features_for(features, model["cards"], variant)
    heads = model["variants"][variant]
    d = forecast(heads["D"], row)
    h = forecast(heads["H"], row) if heads["H"] else None
    return {"remind": d > model["threshold"], "predicted_D": d, "empirical_H": h}


def weighted_fit(rows, cards):
    require(rows, "empty complete-pair training set")
    counts = Counter(r["task_id"] for r in rows)
    weights = [1 / counts[r["task_id"]] for r in rows]
    variants = {}
    d, h = ([r["sample"]["targets"][k] for r in rows] for k in ("D", "H"))
    for variant in ("full", "no_a0"):
        cols = BASE + ACTION if variant == "full" else BASE
        x = [features_for(r["sample"]["features"], cards, variant) for r in rows]
        variants[variant] = {
            "D": stump(x, d, weights, cols),
            "H": stump(x, h, weights, cols) if len(set(h)) == 2 else None,
            "columns": list(cols),
        }
    return variants, weights


def load_training(base, supplement, bank):
    orig = signed(base / "preregistration.json", "plan_id")
    sup = signed(supplement / "preregistration.json", "plan_id")
    release = signed(base / "dataset-v1/dataset.json", "dataset_id")
    rich = read(supplement / "deduplicated-training/summary.json")
    study = signed(supplement / "mechanism/preregistration.json", "study_id")
    bank_doc = signed(bank, "bank_id")
    require(
        release["dataset_id"] == sup["parent_dataset_id"]
        and release["plan_id"] == orig["plan_id"] == sup["parent_plan_id"]
        and rich["plan_id"] == study["plan_id"] == sup["plan_id"]
        and rich["study_id"] == study["study_id"]
        and orig["bank_id"] == sup["bank_id"] == release["bank_id"]
        and bank_doc["bank_id"] == orig["parent_bank_id"]
        and sup["descending_memory_id"] == orig["descending_memory_id"]
        and orig["label_spec"] == sup["label_spec"]
        and sup["label_spec"] == fp.LABELS
        and sup["label_spec"]["m_label"] == DEFAULT_THRESHOLD,
        "incompatible training releases",
    )
    cards = {
        bank_doc["cards"][0]["memory_id"]: "ascending",
        orig["descending_memory_id"]: "descending",
    }
    rows = []
    provenance = []
    seen = {}
    task_origins = {}
    for name, root, plan in (("formal", base, orig), ("supplement", supplement, sup)):
        if name == "formal":
            samples = lines(root / "dataset-v1/train.jsonl")
            evidence = lines(root / "dataset-v1/train-evidence.jsonl")
            ds = release["splits"]["train"]
            expected_s, expected_e = ds["samples_digest"], ds["evidence_digest"]
        else:
            samples = lines(root / "deduplicated-training/all-pairs.jsonl")
            evidence = lines(root / "deduplicated-training/evidence.jsonl")
            expected_s, expected_e = rich["all_pairs_digest"], rich["evidence_digest"]
        require(
            digest(samples) == expected_s
            and digest(evidence) == expected_e
            and len(samples) == len(evidence),
            "training export changed",
        )
        mf = read(root / "train/manifest.json")
        cohort = next(c["tasks"] for c in plan["cohorts"] if c["split"] == "train")
        require(
            mf["split"] == "train"
            and mf["formal_plan_id"] == plan["plan_id"]
            and mf["tasks"] == [r["task"] for r in cohort]
            and mf["actor"] == plan["actor"],
            "training lineage changed",
        )
        tasks = {r["episode_id"]: r for r in cohort}
        for sample, ev in zip(samples, evidence, strict=True):
            cp = ev["checkpoint"].removeprefix("train/")
            eid, turn = cp.split("/")
            require(
                eid in tasks and turn.isdigit() and len(turn) == 3,
                "invalid train checkpoint",
            )
            sid = digest([plan["plan_id"], "train", mf["run_id"], cp, ev["pair_id"]])
            task = tasks[eid]["task"]
            require(
                sid == sample["sample_id"] == ev["sample_id"]
                and digest(task) == tasks[eid]["task_digest"]
                and sample["features"]["instruction"] == task["instruction"]
                and sample["features"]["working_state"]["input"] == task["input"]
                and cards[sample["features"]["memory"]["id"]] == task["family"],
                "sample/checkpoint mismatch",
            )
            t = sample["targets"]
            arms = {
                a: ev["arms"][a]["observed"] if name == "formal" else ev[a]
                for a in ("remind", "silent")
            }
            s, r = arms["silent"], arms["remind"]
            require(
                t["K"] == 1
                and t["hard_label"] == "uncertain"
                and t["hard_loss_weight"] == 0
                and t["continuous_loss_weight"] == 1
                and all(a["success"] in (0, 1) for a in arms.values())
                and math.isclose(t["D"], r["utility"] - s["utility"], abs_tol=1e-9)
                and t["H"] == int(s["success"] == 1 and r["success"] == 0),
                "training target differs from paired arms",
            )
            key = (mf["run_id"], cp)
            if key in seen:
                require(seen[key] == digest(sample), "conflicting duplicate checkpoint")
                continue
            seen[key] = digest(sample)
            task_id = digest(task)
            origin = (mf["run_id"], eid)
            require(
                task_id not in task_origins or task_origins[task_id] == origin,
                "task appears in two source runs",
            )
            task_origins[task_id] = origin
            rows.append(
                {
                    "sample": sample,
                    "task_id": task_id,
                    "source": name,
                    "checkpoint": cp,
                    "run_id": mf["run_id"],
                }
            )
        provenance.append(
            {
                "source": name,
                "plan_id": plan["plan_id"],
                "run_id": mf["run_id"],
                "input_rows": len(samples),
                "samples_digest": expected_s,
                "evidence_digest": expected_e,
            }
        )
    return rows, cards, provenance, orig


def fit(args):
    rows, cards, sources, plan = load_training(
        args.formal_root, args.supplement_root, args.parent_bank
    )
    variants, weights = weighted_fit(rows, cards)
    audit = [
        {
            "sample_id": r["sample"]["sample_id"],
            "task_id": r["task_id"],
            "source": r["source"],
            "run_id": r["run_id"],
            "checkpoint": r["checkpoint"],
            "weight": w,
        }
        for r, w in zip(rows, weights, strict=True)
    ]
    harms = sum(r["sample"]["targets"]["H"] for r in rows)
    model = {
        "schema": "mini-supervised-gate/v1",
        "fit_code": fp.code_digest("gate.py"),
        "bank_id": plan["bank_id"],
        "engine": plan["protocol"],
        "actor_digest": digest(plan["actor"]),
        "cards": cards,
        "variants": variants,
        "threshold": DEFAULT_THRESHOLD,
        "min_leaf_task_weight": MIN_LEAF_TASK_WEIGHT,
        "split_selection": "Depth 1, minimize task-weighted squared loss; fixed column order on ties. Constant fallback.",
        "action_rule": "Research policy: remind iff predicted D exceeds threshold at an eligible, funded checkpoint.",
        "sources": sources,
        "train_ledger_digest": digest(audit),
        "task_ids": sorted({r["task_id"] for r in rows}),
        "sample_count": len(rows),
        "risk_events": harms,
        "risk_status": "uncalibratable_single_class"
        if harms in (0, len(rows))
        else "empirical_uncalibrated",
        "risk_use": "Descriptive only; no deployment risk guarantee.",
        "label_spec": plan["label_spec"],
    }
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "training-ledger.json", audit)
    save(args.out / "model.json", {**model, "model_id": digest(model)})
    print(
        "frozen",
        digest(model),
        "pairs",
        len(rows),
        "tasks",
        len(model["task_ids"]),
        "risk events",
        harms,
    )


def load_model(path):
    m = signed(path, "model_id")
    require(
        m["schema"] == "mini-supervised-gate/v1"
        and m["fit_code"] == fp.code_digest("gate.py")
        and m["train_ledger_digest"]
        == digest(read(path.parent / "training-ledger.json")),
        "model code or training ledger mismatch",
    )
    return m


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("command", choices=["fit"])
    cli.add_argument("--formal-root", type=Path, default=fp.OUTPUT / "cave-formal-01")
    cli.add_argument(
        "--supplement-root", type=Path, default=fp.OUTPUT / "cave-supplement-01"
    )
    cli.add_argument("--parent-bank", type=Path, default=fp.DEFAULT_BANK)
    cli.add_argument("--out", type=Path, required=True)
    fit(cli.parse_args())
