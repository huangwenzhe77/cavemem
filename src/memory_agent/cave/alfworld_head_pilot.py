"""Fit paired gain and harm heads on a separate ALFWorld target cohort."""

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from memory_agent.cave.alfworld_env import MAX_STEPS, sha_file
from memory_agent.cave.alfworld_study_rollout import receipt
from memory_agent.cave.alfworld_transfer_study import (
    audit_pair,
    draft_tokens,
    goal_profile,
    score_tokens,
    similarity,
)
from memory_agent.cave.records import read, save


FEATURES = (
    "checkpoint_step", "remaining_calls", "token_budget", "remaining_tokens", "draft_tokens",
    "menu_size", "shortlist_size", "goal_intent_overlap", "goal_entity_overlap", "original_agrees",
    "source_action_in_shortlist", "strict_contract", "source_verified_gain", "source_repair_steps",
    "source_trajectory_steps", "source_note_words", "recent_action_repeat_count",
)
MIN_CLASS_GAMES = 5
RISK_LIMIT = 0.20


def decision_features(natural, match, card, vocabulary, budget):
    index = match["step"] - 1
    steps = natural["trace"]["steps"]
    row = steps[index]
    draft = row["draft"]
    spent = sum(draft_tokens(step["draft"]) for step in steps[:index + 1])
    target_intent, target_entities = goal_profile(vocabulary, row["goal"])
    source_intent, source_entities = goal_profile(vocabulary, card["source_goal"])
    return [
        index + 1, MAX_STEPS - index - 1, budget or 0, max(0, budget - spent) if budget else 0,
        draft_tokens(draft), len(row["public_menu"]), len(draft["ranked_candidates"]),
        similarity(target_intent, source_intent), similarity(target_entities, source_entities),
        int(match["original"] == match["analogous"]), int(match["source_action_legal_in_shortlist"]),
        int(match["strict_contract"]), card["utility"], card["direct_remaining"],
        len(card["source_commands"]), len(card["text"].split()),
        sum(step["draft"]["action"] == draft["action"] for step in steps[max(0, index - 5):index]),
    ]


def load_cohort(root, study, train_limit=None):
    manifest = read(root / "manifest.json")
    frozen = read(study / "transfer-freeze.json")
    probe = read(root / "probe-freeze.json")
    if (probe["study_sha256"] != sha_file(study / "transfer-freeze.json")
            or probe["cohort_sha256"] != sha_file(root / "manifest.json")):
        raise ValueError("cohort or frozen transfer policy has changed")
    bank_path = study / "transfer-bank.json"
    if sha_file(bank_path) != frozen["bank_sha256"]:
        raise ValueError("memory bank changed after collection")
    cards = {c["id"]: c for c in read(bank_path)["cards"]}
    base = Path(frozen["baseline"])
    ranker = read(base / "ranker.json")
    vocabulary = set(ranker["vocabulary"])
    specs = {s["id"]: s for s in manifest["splits"]["development"]}
    ids = manifest["heads_splits"]
    if (set(ids["train"]).intersection(ids["validation"])
            or set(ids["train"] + ids["validation"]) != set(specs)
            or len({s["template"] for s in specs.values()}) != len(specs)):
        raise ValueError("cohort split is incomplete or shares templates")
    if train_limit is not None:
        if train_limit < 2 or train_limit % 2 or train_limit > len(ids["train"]):
            raise ValueError("interim fit needs an even training count within the registered cohort")
        ids = {"train": ids["train"][:train_limit], "validation": ids["validation"][:train_limit // 2]}
    cohort = {}
    for split, games in ids.items():
        rows = []
        for game in games:
            spec = specs[game]
            natural = read(receipt(root, spec))
            paired = read(receipt(root, spec, "paired"))
            branches = audit_pair(argparse.Namespace(out=root), spec, natural, paired, set(cards))
            for setting, budget in [("calls", None), *frozen["token_budgets"].items()]:
                outcome = paired if budget is None else score_tokens(natural, paired["match"], branches, budget)
                silent, remind = outcome["silent"], outcome["episodic"]
                match = outcome["match"]
                features = decision_features(natural, match, cards[match["card"]], vocabulary, budget) if match else None
                rows.append({
                    "game": game, "setting": setting, "features": features, "strict": bool(match and match["strict_contract"]),
                    "silent": silent, "remind": remind, "D": remind["utility"] - silent["utility"],
                    "H": int(silent["success"] and not remind["success"]),
                })
        cohort[split] = rows
    return cohort


def case_weights(rows):
    counts = Counter(row["game"] for row in rows)
    return torch.tensor([1.0 / counts[row["game"]] for row in rows])


def fit(rows):
    matched = [row for row in rows if row["features"] is not None]
    if not matched:
        raise ValueError("no matched training checkpoints")
    class_games = {label: len({row["game"] for row in matched if row["H"] == label}) for label in (0, 1)}
    fit_risk = min(class_games.values()) >= MIN_CLASS_GAMES
    matrix = torch.tensor([row["features"] for row in matched], dtype=torch.float32)
    gain = torch.tensor([row["D"] for row in matched])
    harm = torch.tensor([row["H"] for row in matched], dtype=torch.float32)
    weights = case_weights(matched)
    weights /= weights.sum()
    center = (weights[:, None] * matrix).sum(dim=0)
    scale = (weights[:, None] * (matrix - center).square()).sum(dim=0).sqrt().clamp_min(0.01)
    inputs = (matrix - center) / scale
    torch.manual_seed(20261122)
    model = nn.Sequential(nn.Linear(len(FEATURES), 8), nn.Tanh(), nn.Linear(8, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)
    for _ in range(300):
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = (weights * (predictions[:, 0] - gain).square()).sum()
        if fit_risk:
            loss += (weights * F.binary_cross_entropy_with_logits(predictions[:, 1], harm, reduction="none")).sum()
        loss.backward()
        optimizer.step()
    return model, center, scale, fit_risk, {
        "matched_rows": len(matched), "matched_games": len({row["game"] for row in matched}),
        "positive_games": class_games[1], "negative_games": class_games[0],
        "gain_baseline": float((weights * gain).sum()), "risk_baseline": float((weights * harm).sum()),
        "strict_matches": sum(row["strict"] for row in matched),
    }


@torch.no_grad()
def evaluate(rows, model, center, scale, fit_risk, training):
    metrics = {}
    estimates = []
    for row in rows:
        matched = row["features"] is not None
        gain = risk = None
        if matched:
            x = torch.tensor(row["features"], dtype=torch.float32)
            predicted = model((x - center) / scale)
            gain = float(predicted[0])
            risk = float(predicted[1].sigmoid()) if fit_risk else None
        estimates.append({"game": row["game"], "setting": row["setting"], "gain": gain, "risk": risk,
                          "D": row["D"], "H": row["H"], "strict": row["strict"]})
        by_setting = metrics.setdefault(row["setting"], {"tasks": 0, "matches": 0, "strict": 0, "policies": {}})
        by_setting["tasks"] += 1
        by_setting["matches"] += int(matched)
        by_setting["strict"] += int(row["strict"])
        policies = {"silent": False, "remind": matched, "gain": matched and gain > 0}
        if fit_risk:
            policies["gain_risk"] = matched and gain > 0 and risk <= RISK_LIMIT
        for policy, remind in policies.items():
            observed = row["remind"] if remind else row["silent"]
            stats = by_setting["policies"].setdefault(policy, Counter())
            stats.update({"reminders": int(remind), "wins": observed["success"], "tokens": observed["tokens"],
                          "seconds": observed["seconds"], "utility": observed["utility"],
                          "harms": int(remind and row["H"]),
                          "rescues": int(remind and observed["success"] and not row["silent"]["success"])})
    matched_estimates = [row for row in estimates if row["gain"] is not None]
    errors = {}
    if matched_estimates:
        weights = case_weights(matched_estimates)
        weights /= weights.sum()

        def loss(metric, target, baseline=None, square=False):
            delta = torch.tensor([(r[metric] if baseline is None else baseline) - r[target] for r in matched_estimates])
            return float((weights * (delta.square() if square else delta.abs())).sum())

        errors = {"gain_mae": loss("gain", "D"), "gain_constant_mae": loss("gain", "D", training["gain_baseline"]),
                  "harm_rows": sum(r["H"] for r in matched_estimates),
                  "harm_games": len({r["game"] for r in matched_estimates if r["H"]})}
        if fit_risk:
            errors.update(risk_brier=loss("risk", "H", square=True),
                          risk_constant_brier=loss("risk", "H", training["risk_baseline"], square=True))
    return {"by_budget": metrics, "prediction_error": errors, "estimates": estimates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "study"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--interim-train-games", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cohort = load_cohort(args.out, args.study, args.interim_train_games)
    model, center, scale, fit_risk, training = fit(cohort["train"])
    validation = evaluate(cohort["validation"], model, center, scale, fit_risk, training)
    weights = {key: value.tolist() for key, value in model.state_dict().items()}
    if not fit_risk:
        weights["2.weight"] = weights["2.weight"][:1]
        weights["2.bias"] = weights["2.bias"][:1]
    dest = args.out if args.interim_train_games is None else args.out / f"interim-{args.interim_train_games}"
    save(dest / "heads-model.json", {"features": FEATURES, "width": 8, "activation": "tanh",
                                     "center": center.tolist(), "scale": scale.tolist(), "weights": weights,
                                     "risk_available": fit_risk, "risk_limit": RISK_LIMIT,
                                     "train_ids": sorted({row["game"] for row in cohort["train"]}),
                                     "cohort_sha256": sha_file(args.out / "manifest.json"),
                                     "transfer_sha256": sha_file(args.study / "transfer-freeze.json")})
    save(dest / "heads-report.json", {"risk_available": fit_risk, "training": training, "validation": validation})
    print("HEADS_FIT", training, "risk_available", fit_risk, flush=True)
    print("HEADS_VALIDATION", validation["prediction_error"], flush=True)


if __name__ == "__main__":
    main()
