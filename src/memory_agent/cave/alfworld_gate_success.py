"""Lock a success-first gate on validation data and score it on sealed paired outcomes."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from memory_agent.cave.alfworld_env import sha_file
from memory_agent.cave.alfworld_gate_refit import paired_rows, policy_report
from memory_agent.cave.alfworld_gate_test import interval
from memory_agent.cave.alfworld_gate_tune import CONTEXT_FEATURES, FEATURES, SEED, SETTINGS, context_features, predict
from memory_agent.cave.alfworld_head_pilot import decision_features
from memory_agent.cave.alfworld_study_pairs import context_applies
from memory_agent.cave.alfworld_study_rollout import receipt
from memory_agent.cave.alfworld_transfer_study import choose_transfer
from memory_agent.cave.records import read, save


def verify(args):
    refit_registry = read(args.refit / "cohort.json")
    selection_file = args.out / "frozen-selection.json"
    selection = read(selection_file)
    transfer_file = args.study / "transfer-freeze.json"
    transfer = read(transfer_file)
    seal_file = args.refit / "sealed-test.json"
    seal = read(seal_file)
    labels_file = args.refit / "paired-labels.json"
    if (sha_file(seal_file) != selection["seal_sha256"]
            or sha_file(transfer_file) != selection["transfer_sha256"]
            or sha_file(labels_file) != selection["labels_sha256"]):
        raise ValueError("validation or test registry has changed")
    return refit_registry, selection, transfer, seal


def shortlist(args, cohort, selection, transfer):
    base = read(args.refit / "paired-labels.json")["validation"]
    manifest = read(Path(transfer["baseline"]) / "manifest.json")
    enriched = paired_rows(args.study, transfer, manifest, cohort, context_features)["validation"]
    data = {FEATURES: base, CONTEXT_FEATURES: enriched}
    models = {k: m for k, m in selection["models"].items() if m["spec"]["width"] > 0}
    frozen = {}
    decisions = np.zeros(len(base), dtype=bool)
    for setting in SETTINGS:
        indexes = np.array([i for i, r in enumerate(base) if r["setting"] == setting])
        part = [base[i] for i in indexes]
        _, g, size = np.unique([r["template"] for r in part], return_inverse=True, return_counts=True)
        groups = len(size)
        boot = np.random.default_rng(SEED).multinomial(groups, np.full(groups, 1/groups), size=1024) / groups
        weights = boot[:, g] / size[g]
        win = np.array([r["remind"]["success"] - r["silent"]["success"] for r in part])
        harm, extra = (np.array([r[k] for r in part]) for k in ("H", "extra"))
        best = (0, 0, 0)
        frozen[setting] = {"always_silent": True}
        for key, model in models.items():
            feature_rows = [data[tuple(model["features"])][i] for i in indexes]
            mask = np.array([r["features"] is not None for r in feature_rows])
            forecasts = np.full((len(part), 2), np.nan)
            forecasts[mask] = predict([r for r in feature_rows if r["features"] is not None], model)
            for tau in (0., .01, .03, .06, .12, .20, .25, .35, .50):
                for ceiling in (.025, .05, .1, .2, .35, 1.):
                    act = mask & (forecasts[:, 0] > tau) & (forecasts[:, 1] <= ceiling)
                    score = (int(win @ act), -int(extra @ act), -int(harm @ act))
                    if score <= best:
                        continue
                    h95 = float(np.quantile(weights @ (harm * act), .95))
                    c95 = float(np.quantile(weights @ (extra * act), .95))
                    if (h95 > cohort["limits"]["policy_risk_delta"]
                            or c95 > cohort["limits"]["policy_mean_extra_tokens_budget"]):
                        continue
                    best = score
                    frozen[setting] = {"model": key, "gain": tau, "risk": ceiling,
                                       "validation_success_change": score[0], "validation_token_change": -score[1],
                                       "validation_risk_upper": h95, "validation_extra_tokens_upper": c95}
                    decisions[indexes] = act
    return frozen, policy_report(base, decisions)


def score(args, cohort, selection, transfer, seal, rules):
    if not (args.out / "test-report.json").is_file():
        raise ValueError("the sealed cohort must complete before scoring")
    base = Path(transfer["baseline"])
    vocab = set(read(base / "ranker.json")["vocabulary"])
    cards = read(args.study / "transfer-bank.json")["cards"]
    actor = SimpleNamespace(ranker=SimpleNamespace(vocabulary=vocab))
    builders = {FEATURES: decision_features, CONTEXT_FEATURES: context_features}
    scored, decisions = [], []
    for spec in seal["splits"]["test"]:
        rows = read(receipt(args.out, spec, "scored"))
        natural = read(receipt(args.out, spec))
        # Recover the shared anchor without using the test continuation to choose a decision.
        if any(r["matched"] for r in rows):
            anchor, card, analogous = choose_transfer(actor, natural["trace"]["steps"], cards)
            match = {"step": len(anchor["prefix"]) + 1, "original": anchor["draft"]["action"], "analogous": analogous,
                     "strict_contract": context_applies(actor, anchor, analogous, card),
                     "source_action_legal_in_shortlist": card["source_action"] in anchor["draft"]["allowed"]}
        for row in rows:
            policy = rules[row["setting"]]
            decision = False
            if row["matched"] and not policy.get("always_silent", False):
                model = selection["models"][policy["model"]]
                feature_builder = builders[tuple(model["features"])]
                x = feature_builder(natural, match, card, vocab, cohort["budgets"].get(row["setting"]))
                gain, risk = predict([{"features": x, "setting": row["setting"]}], model)[0]
                decision = bool(gain > policy["gain"] and risk <= policy["risk"])
            scored.append(row)
            decisions.append(decision)
    policy = policy_report(scored, decisions)
    evidence = {}
    for setting in SETTINGS:
        part = [(r, a) for r, a in zip(scored, decisions, strict=True) if r["setting"] == setting]
        data = [r for r, _ in part]
        delta_wins = [int(a) * (r["remind"]["success"] - r["silent"]["success"]) for r, a in part]
        delta_tokens = [int(a) * r["extra"] for r, a in part]
        evidence[setting] = {
            "win_delta_vs_silent": interval(data, delta_wins),
            "token_delta_vs_silent": interval(data, delta_tokens),
            "utility_delta_vs_silent": interval(data, [int(a) * r["D"] for r, a in part]),
            "saved_harms": sum(r["H"] for r, a in part if not a),
            "missed_rescues": sum(r["remind"]["success"] > r["silent"]["success"] for r, a in part if not a),
            "dual_success_avoided_extra_tokens": sum(
                r["extra"] for r, a in part if not a and r["extra"] > 0 and r["silent"]["success"] and r["remind"]["success"]
            ),
        }
    return {"policy": policy, "evidence": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("lock", "score"))
    for name in ("out", "refit", "study"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    cohort, selection, transfer, seal = verify(args)
    lock_path = args.out / "success-first-freeze.json"
    if args.stage == "lock":
        if lock_path.exists() or (args.out / "test-report.json").exists():
            raise ValueError("the success-first rule can only be locked once, before opening the test report")
        rules, validation = shortlist(args, cohort, selection, transfer)
        save(lock_path, {"rules": rules, "validation": validation,
                         "selection_sha256": sha_file(args.out / "frozen-selection.json"),
                         "test_freeze_sha256": sha_file(args.out / "test-freeze.json"),
                         "code_sha256": sha_file(Path(__file__)),
                         "criterion": "validation successes first, tokens second, within validation risk/extra-token limits"})
        print("SUCCESS_FIRST_LOCKED", rules, flush=True)
    else:
        lock = read(lock_path)
        if (lock["selection_sha256"] != sha_file(args.out / "frozen-selection.json")
                or lock["test_freeze_sha256"] != sha_file(args.out / "test-freeze.json")
                or lock["code_sha256"] != sha_file(Path(__file__))):
            raise ValueError("changed success-first policy or test release")
        report = score(args, cohort, selection, transfer, seal, lock["rules"])
        save(args.out / "success-first-report.json", report)
        print("SUCCESS_FIRST_COMPLETE", len(seal["splits"]["test"]), flush=True)


if __name__ == "__main__":
    main()
