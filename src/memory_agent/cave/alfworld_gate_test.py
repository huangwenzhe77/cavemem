"""Evaluate locked gate models on the sealed ALFWorld templates."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from memory_agent.cave.alfworld_env import Actor, sha_file
from memory_agent.cave.alfworld_gate_refit import policy_report
from memory_agent.cave.alfworld_gate_tune import (
    CONTEXT_FEATURES, FEATURES, SEED, SETTINGS, config_for, context_features, predict,
)
from memory_agent.cave.alfworld_head_pilot import decision_features
from memory_agent.cave.alfworld_ranker import RankedActor, examples, load
from memory_agent.cave.alfworld_study_pairs import cached, context_applies
from memory_agent.cave.alfworld_study_rollout import receipt, rollout
from memory_agent.cave.alfworld_transfer_study import (
    NEUTRAL, audit_pair, choose_transfer, continuing_advice, episodic_note, score_tokens,
)
from memory_agent.cave.records import read, save


def frozen_inputs(args):
    transfer_path = args.study / "transfer-freeze.json"
    transfer = read(transfer_path)
    selection_path = args.out / "frozen-selection.json"
    selection = read(selection_path)
    seal_path = args.refit / "sealed-test.json"
    seal = read(seal_path)
    registry = read(args.refit / "cohort.json")
    base = Path(transfer["baseline"])
    config = read(base / "actor.json")
    bank_path = args.study / "transfer-bank.json"
    checks = [
        (seal_path, selection["seal_sha256"]), (seal_path, registry["sealed_test_sha256"]),
        (transfer_path, selection["transfer_sha256"]), (bank_path, transfer["bank_sha256"]),
        (args.config, seal["config_sha256"]), (base / "actor.json", transfer["actor_sha256"]),
        (base / "ranker.pt", config["ranker"]),
        *[(Path(__file__).with_name(name), checksum) for name, checksum in transfer["code"].items()],
    ]
    if (not seal["splits"]["test"] or len(seal["splits"]["test"]) != 150
            or str(args.weights.resolve()) != config["actor"] or any(sha_file(p) != h for p, h in checks)):
        raise ValueError("test input differs from the sealed actor, bank, or game cohort")
    frozen = {"manifest_sha256": sha_file(seal_path), "selection_sha256": sha_file(selection_path),
              "transfer_sha256": sha_file(transfer_path), "budgets": transfer["token_budgets"],
              "code": {name: sha_file(Path(__file__).with_name(name)) for name in (
                  "alfworld_gate_test.py", "alfworld_gate_tune.py", "alfworld_gate_refit.py", "alfworld_head_pilot.py",
              )},
              "estimand": "single gate decision at the first eligible archived-transfer match"}
    path = args.out / "test-freeze.json"
    if path.exists() and read(path) != frozen:
        raise ValueError("can only resume with the same test protocol")
    save(path, frozen)
    args.exposure = transfer["exposure"]
    args.representation = transfer["representation"]
    ranker = load(base)
    actor = RankedActor(Actor(args.weights), ranker, config["top"], config.get("repeat_aware", False),
                        examples(base, ranker) if config.get("examples") else (), config["indexed"])
    models = {
        "mlp_mae": selection["models"][selection["selected_mlp"]],
        "mlp_gate": selection["models"][selection["selected_gate"]],
        "linear": selection["models"]["0"],
    }
    return seal["splits"]["test"], read(bank_path)["cards"], actor, models, transfer["token_budgets"]


def test_pair(args, spec, actor, cards, models, budgets):
    natural = cached(receipt(args.out, spec), lambda: rollout(args, spec, actor))
    if natural["spec"] != spec or natural["extra_draft"] is not None:
        raise ValueError(f"changed natural trajectory for {spec['id']}")
    choice = choose_transfer(actor, natural["trace"]["steps"], cards)
    silent = natural["metrics"]
    result = {"spec": spec["id"], "match": None, "silent": silent, "episodic": silent, "neutral": silent}
    forecasts = {}
    if choice is not None:
        anchor, card, analogous = choice
        step = len(anchor["prefix"]) + 1
        match = {"step": step, "state": anchor["state"], "source": card["source"], "card": card["id"],
                 "original": anchor["draft"]["action"], "analogous": analogous,
                 "source_action_legal_in_shortlist": card["source_action"] in anchor["draft"]["allowed"],
                 "strict_contract": context_applies(actor, anchor, analogous, card), "evidence": {}}
        result["match"] = match
        vocabulary = actor.ranker.vocabulary
        for setting, budget in [("calls", None), *budgets.items()]:
            features = {
                FEATURES: decision_features(natural, match, card, vocabulary, budget),
                CONTEXT_FEATURES: context_features(natural, match, card, vocabulary, budget),
            }
            forecasts[setting] = {}
            for name, model in models.items():
                row = {"setting": setting, "features": features[tuple(model["features"])]}
                estimate = predict([row], model)[0]
                forecasts[setting][name] = {"gain": float(estimate[0]), "risk": float(estimate[1]),
                                             "remind": bool(config_for(model, row, estimate))}
        preview_path = receipt(args.out, spec, "locked-decisions")
        if preview_path.exists() and read(preview_path) != forecasts:
            raise ValueError("pre-intervention predictions changed")
        if not preview_path.exists():
            save(preview_path, forecasts)
        notice = episodic_note(card, anchor["draft"]["allowed"], args.representation)
        ongoing = continuing_advice(card, step - 1, args)
        episode_path = receipt(args.out, spec, "episodic")
        neutral_path = receipt(args.out, spec, "neutral-first")
        episode = cached(episode_path, lambda: rollout(
            args, spec, actor, anchor=anchor, reminder=notice, continuation_reminder=ongoing,
        ))
        # Only the first neutral re-draft cost is used by the unchanged affordability rule.
        neutral = cached(neutral_path, lambda: rollout(args, spec, actor, anchor=anchor, reminder=NEUTRAL, cap=step + 1))
        for arm, path, run in (("episodic", episode_path, episode), ("neutral", neutral_path, neutral)):
            result[arm] = run["metrics"]
            match["evidence"][arm] = {"file": path.name, "sha256": sha_file(path)}
            match[f"{arm}_first_action"] = run["trace"]["steps"][step - 1]["draft"]["action"]
    branches = audit_pair(args, spec, natural, result, {c["id"] for c in cards})
    answers = []
    for setting, budget in [("calls", None), *budgets.items()]:
        observed = result if budget is None else score_tokens(natural, result["match"], branches, budget)
        matched = observed["match"] is not None
        s, r = observed["silent"], observed["episodic"]
        answers.append({"game": spec["id"], "template": spec["template"], "family": spec["family"], "setting": setting,
                        "matched": matched, "silent": s, "remind": r,
                        "extra": r["tokens"] - s["tokens"], "D": r["utility"] - s["utility"],
                        "H": int(s["success"] and not r["success"]),
                        "forecast": forecasts[setting] if matched else {}})
    return answers


def interval(rows, values):
    grouped = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        grouped[row["template"]].append(value)
    estimates = np.asarray([np.mean(v) for v in grouped.values()])
    samples = np.random.default_rng(SEED).integers(len(estimates), size=(2000, len(estimates)))
    return {"mean": float(estimates.mean()),
            "ci95": np.quantile(estimates[samples].mean(axis=1), [.025, .975]).tolist()}


def average_precision(harm, risk):
    if not harm.any():
        return None
    order = np.argsort(-risk, kind="stable")
    y = harm[order]
    scores = risk[order]
    boundaries = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    positive = y.cumsum()[boundaries]
    return float(np.diff(np.r_[0, positive]) @ (positive / (boundaries + 1)) / positive[-1])


def summarize(rows, models):
    matched = [r for r in rows if r["matched"]]
    reports = {}
    policies = {"silent": [False] * len(rows), "always_remind": [r["matched"] for r in rows]}
    for name, model in models.items():
        policies[name] = [bool(r["matched"] and r["forecast"][name]["remind"]) for r in rows]
        predictions = {}
        for setting in ("overall", *SETTINGS):
            part = [r for r in matched if setting == "overall" or r["setting"] == setting]
            if not part:
                continue
            actual = np.array([r["D"] for r in part])
            gain = np.array([r["forecast"][name]["gain"] for r in part])
            risk = np.array([r["forecast"][name]["risk"] for r in part])
            harm = np.array([r["H"] for r in part])
            budget_median = np.array([model["baseline"][r["setting"]]["median_gain"] for r in part])
            budget_prior = np.array([model["baseline"][r["setting"]]["risk"] for r in part])
            error = abs(actual - gain)
            baseline_error = abs(actual - budget_median)
            predictions[setting] = {"rows": len(part), "mae": float(error.mean()),
                                    "budget_median_mae": float(baseline_error.mean()),
                                    "mae_improvement": interval(part, baseline_error - error),
                                    "risk_brier": float(((risk - harm)**2).mean()),
                                    "budget_prior_brier": float(((budget_prior - harm)**2).mean()),
                                    "risk_average_precision": average_precision(harm, risk),
                                    "risk_prevalence": float(harm.mean())}
        reports[name] = predictions
    policy = {name: policy_report(rows, act) for name, act in policies.items()}
    effects = {}
    for name, acts in policies.items():
        if name == "silent":
            continue
        effects[name] = {}
        for setting in ("overall", *SETTINGS):
            pairs = [(r, act) for r, act in zip(rows, acts, strict=True) if setting == "overall" or r["setting"] == setting]
            part = [r for r, _ in pairs]
            effects[name][setting] = {
                "gain_vs_silent": interval(part, [r["D"] * act for r, act in pairs]),
                "gain_vs_always_remind": interval(part, [r["D"] * (act - r["matched"]) for r, act in pairs]),
            }
    return {"games": len({r["game"] for r in rows}), "matches": {s: sum(r["matched"] and r["setting"] == s for r in rows) for s in SETTINGS},
            "prediction": reports, "policies": policy, "utility_effect": effects}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "refit", "study", "data", "config", "weights"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args()
    specs, cards, actor, models, budgets = frozen_inputs(args)
    rows = []
    for i, spec in enumerate(specs, 1):
        path = receipt(args.out, spec, "scored")
        all_settings = cached(path, lambda: test_pair(args, spec, actor, cards, models, budgets))
        if {r["setting"] for r in all_settings} != set(SETTINGS) or any(r["game"] != spec["id"] for r in all_settings):
            raise ValueError("unpaired or mislabeled sealed task")
        rows.extend(all_settings)
        print(f"TEST_PROGRESS {i}/{len(specs)} {spec['id']}", flush=True)
        save(args.out / "test-progress.json", {"completed": i, "total": len(specs)})
    report = summarize(rows, models)
    save(args.out / "test-report.json", {**report, "test_freeze_sha256": sha_file(args.out / "test-freeze.json")})
    print("SEALED_TEST_COMPLETE", report["games"], flush=True)


if __name__ == "__main__":
    main()
