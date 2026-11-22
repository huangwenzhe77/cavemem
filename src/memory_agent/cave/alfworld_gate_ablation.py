"""Compare the learned decision to memory-only and proposal-blind controllers."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from memory_agent.cave.alfworld_env import sha_file
from memory_agent.cave.alfworld_gate_refit import policy_report
from memory_agent.cave.alfworld_gate_test import interval
from memory_agent.cave.alfworld_gate_tune import CONTEXT_FEATURES, SETTINGS
from memory_agent.cave.alfworld_stage_policy import decide, fit, predict, select
from memory_agent.cave.alfworld_stage_test import PRIMARY
from memory_agent.cave.records import read, save


PROPOSAL_FIELDS = {"draft_tokens", "original_agrees", "recent_action_repeat_count",
                   "analogous_action_logit_gap", "original_top_two_logit_gap", "candidate_entropy"}


def mask(rows, columns):
    result = []
    for r in rows:
        if r["features"] is None:
            result.append(r)
        else:
            result.append({**r, "features": [v if i in columns else 0 for i, v in enumerate(r["features"])]})
    return result


def lock(args, full, source):
    if (args.test / "test-report.json").exists() or (args.test / "ablation-freeze.json").exists():
        raise ValueError("ablation must be locked prior to opening the test report")
    data_path = args.development / "paired-labels.json"
    if sha_file(data_path) != full["dataset_sha256"]:
        raise ValueError("incompatible ablation data")
    data = read(data_path)
    names = (*CONTEXT_FEATURES, *source["memory_order"])
    if source["selection"] != "legacy-wide-persistent" or len(names) != len(full["model"]["center"]):
        raise ValueError("unexpected feature list for source and action controls")
    columns = {
        "memory_only": set(range(len(CONTEXT_FEATURES), len(names))) | {names.index("token_budget")},
        "without_proposal": {i for i, name in enumerate(names) if name not in PROPOSAL_FIELDS},
    }
    frozen = {}
    for name, fields in columns.items():
        train = mask([r for r in data["train"] if r["features"] is not None], fields)
        validation = mask(data["validation"], fields)
        model = fit(train, full["model"]["width"], full["model"]["penalty"])
        forecast = predict(validation, model)
        rules = select(validation, forecast)
        frozen[name] = {"columns": sorted(fields), "model": model, "rules": rules,
                        "validation": policy_report(validation, decide(validation, forecast, rules))}
    save(args.test / "ablation-freeze.json", {"full_model_sha256": sha_file(args.development / "model.json"),
                                               "test_manifest_sha256": sha_file(args.test / "sealed-test.json"),
                                               "code_sha256": sha_file(Path(__file__)), "comparators": frozen})
    print("GATING_ABLATIONS_LOCKED", list(frozen), flush=True)


def score(args, full, source):
    frozen = read(args.test / "ablation-freeze.json")
    if (not (args.test / "test-report.json").exists()
            or frozen["full_model_sha256"] != sha_file(args.development / "model.json")
            or frozen["test_manifest_sha256"] != sha_file(args.test / "sealed-test.json")
            or frozen["code_sha256"] != sha_file(Path(__file__))):
        raise ValueError("test is incomplete, or ablation was changed after registration")
    specs = read(args.test / "sealed-test.json")["splits"]["test"]
    rows = [r for spec in specs for r in read(args.test / "stage-test" / spec["id"] / "scored.json")["rows"]]
    action = decide(rows, predict(rows, full["model"]), full["rules"])
    comparison = {}
    for name, spec in frozen["comparators"].items():
        alternate_rows = mask(rows, set(spec["columns"]))
        alternate = decide(alternate_rows, predict(alternate_rows, spec["model"]), spec["rules"])
        part = [(r, int(a) - int(b)) for r, a, b in zip(rows, action, alternate, strict=True)
                if r["setting"] == PRIMARY]
        evidence = [r for r, _ in part]
        comparison[name] = {
            "policy": policy_report(rows, alternate),
            "p50_success_gain_full_vs_control": interval(evidence, [d * (r["R"] - r["H"]) for r, d in part]),
            "p50_token_saving_full_vs_control": interval(evidence, [-d * r["extra"] for r, d in part]),
        }
    eligible = defaultdict(list)
    by_budget = {}
    for setting in SETTINGS:
        items = [(r, a) for r, a in zip(rows, action, strict=True) if r["setting"] == setting]
        for row, remind in items:
            if row["features"] is None:
                continue
            index = int(np.argmax(row["features"][-len(source["memory_order"]):]))
            card = source["memory_order"][index]
            eligible[setting, card].append({"game": row["game"], "remind": remind, "R": row["R"],
                                            "H": row["H"], "extra_tokens": row["extra"]})
        by_budget[setting] = {
            "rescues_kept": sum(r["R"] for r, a in items if a),
            "rescues_missed": sum(r["R"] for r, a in items if not a),
            "harms_blocked": sum(r["H"] for r, a in items if not a),
            "harms_incurred": sum(r["H"] for r, a in items if a),
            "tokens_saved_by_silence_in_dual_success": sum(
                max(r["extra"], 0) for r, a in items if not a and r["silent"]["success"] and r["remind"]["success"]
            ),
        }
    contrasts = [{"setting": setting, "memory": card, "rows": part}
                 for (setting, card), part in eligible.items()
                 if {r["remind"] for r in part} == {False, True}]
    save(args.test / "ablation-report.json", {"comparators": comparison, "effect_counts": by_budget,
                                               "same_memory_different_decisions": contrasts})
    print("GATING_ABLATIONS_COMPLETE", by_budget[PRIMARY], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("lock", "score"))
    for name in ("development", "test"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    full = read(args.development / "model.json")
    source = read(args.development / "collector-freeze.json")
    (lock if args.stage == "lock" else score)(args, full, source)


if __name__ == "__main__":
    main()
