"""Select a regularized reminder predictor without reading the sealed test outcomes."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from memory_agent.cave.alfworld_env import sha_file
from memory_agent.cave.alfworld_gate_refit import paired_rows, policy_report
from memory_agent.cave.alfworld_head_pilot import FEATURES, decision_features
from memory_agent.cave.alfworld_ranker import entities, words
from memory_agent.cave.alfworld_transfer_study import episodic_note
from memory_agent.cave.records import read, save


SEED = 262735
SETTINGS = ("calls", "p50", "p75", "p90")
SHRINKAGE = (.25, .50, 1.)
CONTEXT_FEATURES = FEATURES + (
    "candidate_rank", "analogous_action_logit_gap", "original_top_two_logit_gap", "candidate_entropy",
    "reminder_words", "reminder_to_remaining_tokens", "visible_goal_entity_count",
    "observation_words", "same_shape_prior_occurrences",
)


def context_features(natural, match, card, vocabulary, budget):
    result = decision_features(natural, match, card, vocabulary, budget)
    trace = natural["trace"]["steps"]
    row = trace[match["step"] - 1]
    draft = row["draft"]
    logits = np.asarray(draft["choice_logits"], dtype=float)
    weights = np.exp(logits - logits.max())
    probabilities = weights / weights.sum()
    open_action = draft["allowed"].index(match["analogous"])
    top = sorted(logits, reverse=True)
    note = episodic_note(card, draft["allowed"], "sequence-trajectory")
    terms = set(words(row["goal"])) & vocabulary
    visible = {noun for noun, _ in entities(row["observation"])}
    return [
        *result, open_action, float(logits.max() - logits[open_action]),
        float(top[0] - top[1]) if len(top) > 1 else 0.,
        float(-probabilities @ np.log(np.maximum(probabilities, 1e-12))),
        len(note.split()), len(note.split()) / max(1, result[3]) if budget else 0.,
        len(terms & visible), len(row["observation"].split()),
        sum(r["draft"]["action"].split()[0] == match["analogous"].split()[0] for r in trace[:match["step"] - 1]),
    ]


def weights(rows):
    game_size = Counter(r["game"] for r in rows)
    mass = np.array([1 / game_size[r["game"]] for r in rows])
    return mass / mass.sum()


def anchors(rows, length=False):
    result = {}
    for setting in SETTINGS:
        part = [r for r in rows if r["setting"] == setting]
        gain = [r["D"] for r in part]
        result[setting] = {
            "median_gain": float(np.median(gain)), "mean_gain": float(np.mean(gain)),
            "risk": (sum(r["H"] for r in part) + 1) / (len(part) + 2),
        }
        if length:
            col = CONTEXT_FEATURES.index("reminder_words")
            costs = np.array([r["features"][col] for r in part])
            edges = np.unique(np.quantile(costs, [.25, .50, .75]))
            buckets = np.searchsorted(edges, costs, side="right")
            median = result[setting]["median_gain"]
            offsets = []
            for i in range(len(edges) + 1):
                labels = np.asarray(gain)[buckets == i]
                offsets.append(float(len(labels) / (len(labels) + 20) * (np.median(labels) - median))
                               if len(labels) else 0.)
            result[setting].update(length_edges=edges.tolist(), length_offsets=offsets)
    return result


def gain_anchor(rows, anchors):
    values = []
    for row in rows:
        base = anchors[row["setting"]]
        gain = base["median_gain"]
        if "length_edges" in base:
            length = row["features"][CONTEXT_FEATURES.index("reminder_words")]
            bucket = np.searchsorted(base["length_edges"], length, side="right")
            gain += base["length_offsets"][bucket]
        values.append(gain)
    return np.asarray(values)


def normalize(rows, center, scale):
    return np.clip((np.array([r["features"] for r in rows]) - center) / scale, -4, 4)


def fit(rows, spec):
    base = anchors(rows, spec.get("length_prior", False))
    w = weights(rows)
    data = np.array([r["features"] for r in rows])
    center = w @ data
    scale = np.maximum(np.sqrt(w @ (data - center)**2), .01)
    x = torch.tensor(normalize(rows, center, scale), dtype=torch.float32)
    gain_base = torch.tensor(gain_anchor(rows, base), dtype=x.dtype)
    risk_base = torch.tensor([base[r["setting"]]["risk"] for r in rows], dtype=x.dtype)
    risk_logit = torch.logit(risk_base)
    gain = torch.tensor([r["D"] for r in rows], dtype=x.dtype)
    harm = torch.tensor([r["H"] for r in rows], dtype=x.dtype)
    w = torch.tensor(w, dtype=x.dtype)
    torch.manual_seed(SEED)
    width = spec["width"]
    size = x.shape[1]
    model = nn.Sequential(nn.Linear(size, width), nn.Tanh(), nn.Linear(width, 2)) if width else nn.Linear(size, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.012, weight_decay=0)
    for _ in range(220):
        optimizer.zero_grad()
        pred = model(x)
        error = pred[:, 0] + gain_base - gain
        if spec["loss"] == "mse":
            gain_loss = error.square()
        else:
            gain_loss = 2 * F.huber_loss(error, torch.zeros_like(error), reduction="none", delta=.15)
        risk_loss = F.binary_cross_entropy_with_logits(pred[:, 1] + risk_logit, harm, reduction="none")
        penalty = sum(p.square().sum() for p in model.parameters())
        loss = w @ (gain_loss + risk_loss) + spec["l2"] * penalty
        loss.backward()
        optimizer.step()
    if width:
        layers = [[m.weight.detach().tolist(), m.bias.detach().tolist()] for m in (model[0], model[2])]
    else:
        layers = [[model.weight.detach().tolist(), model.bias.detach().tolist()]]
    return {"spec": spec, "center": center.tolist(), "scale": scale.tolist(), "layers": layers, "baseline": base}


def predict(rows, model, shrinkage=None):
    if not rows:
        return np.empty((0, 2))
    if shrinkage is None:
        shrinkage = model["shrinkage"]
    x = normalize(rows, model["center"], model["scale"])
    for i, (matrix, bias) in enumerate(model["layers"]):
        if i:
            x = np.tanh(x)
        x = x @ np.asarray(matrix).T + bias
    baseline = model["baseline"]
    risks = np.array([baseline[r["setting"]]["risk"] for r in rows])
    gain = shrinkage * x[:, 0]
    logits = shrinkage * x[:, 1]
    if not model.get("absolute", False):
        gain += gain_anchor(rows, baseline)
        logits += np.log(risks / (1 - risks))
    return np.column_stack((gain, np.exp(-np.logaddexp(0, -logits))))


def prediction_report(rows, forecast, baseline):
    gain, harm = (np.array([r[key] for r in rows]) for key in ("D", "H"))
    if isinstance(baseline, dict):
        baseline = [baseline[r["setting"]] for r in rows]
    base_median, base_mean, base_risk = (
        np.array([r[k] for r in baseline]) for k in ("median_gain", "mean_gain", "risk")
    )
    w = weights(rows)
    return {
        "gain_mae": float(w @ np.abs(forecast[:, 0] - gain)),
        "gain_rmse": float(np.sqrt(w @ (forecast[:, 0] - gain)**2)),
        "budget_median_mae": float(w @ np.abs(base_median - gain)),
        "budget_mean_mae": float(w @ np.abs(base_mean - gain)),
        "risk_brier": float(w @ (forecast[:, 1] - harm)**2),
        "budget_prior_brier": float(w @ (base_risk - harm)**2),
        "harm_games": len({r["game"] for r in rows if r["H"]}), "rows": len(rows),
    }


def cross_validate(rows, use_length):
    templates = sorted({r["template"] for r in rows})
    np.random.default_rng(SEED).shuffle(templates)
    fold_of = {template: i % 4 for i, template in enumerate(templates)}
    parts = []
    baselines = [None] * len(rows)
    for fold in range(4):
        split = [i for i, row in enumerate(rows) if fold_of[row["template"]] == fold]
        part = [row for row in rows if fold_of[row["template"]] != fold]
        parts.append((split, part))
        baseline = anchors(part)
        for i in split:
            baselines[i] = baseline[rows[i]["setting"]]
    forecasts = {}
    scenarios = [("mse", False), ("huber", False)]
    if use_length:
        scenarios.append(("huber", True))
    for width in (0, 8, 16):
        for loss, length in scenarios:
            for l2 in (.003, .03, .10):
                spec = {"width": width, "loss": loss, "l2": l2, "length_prior": length}
                out = {s: np.empty((len(rows), 2)) for s in SHRINKAGE}
                for split, part in parts:
                    model = fit(part, spec)
                    for s in SHRINKAGE:
                        out[s][split] = predict([rows[i] for i in split], model, s)
                for s in SHRINKAGE:
                    forecasts[width, loss, l2, s, length] = out[s]
    reports = {}
    for name, values in forecasts.items():
        reports[name] = prediction_report(rows, values, baselines)
    ranking = sorted(reports, key=lambda name: (reports[name]["gain_mae"], reports[name]["risk_brier"], name))
    winners = {str(width): next(name for name in ranking if name[0] == width) for width in (0, 8, 16)}
    for width in (8, 16):
        candidates = [name for name in ranking if name[0] == width and name[1] == "mse"]
        winners[f"{width}-expected"] = min(candidates, key=lambda name: reports[name]["gain_rmse"])
    serialized = [{"width": w, "loss": loss, "l2": l2, "shrinkage": s, "length_prior": length,
                   **reports[w, loss, l2, s, length]} for w, loss, l2, s, length in ranking]
    return winners, serialized


def threshold(rows, forecast, limits):
    decisions = np.zeros(len(rows), dtype=bool)
    by_budget = {}
    for setting in SETTINGS:
        indexes = np.array([i for i, row in enumerate(rows) if row["setting"] == setting])
        part = [rows[i] for i in indexes]
        values = forecast[indexes]
        matched = np.array([row["features"] is not None for row in part])
        _, group, size = np.unique([r["template"] for r in part], return_inverse=True, return_counts=True)
        n = len(size)
        boot = np.random.default_rng(SEED).multinomial(n, np.full(n, 1/n), size=1024) / n
        w = boot[:, group] / size[group]
        gain, harm, extra = (np.array([r[key] for r in part]) for key in ("D", "H", "extra"))
        selected = {"always_silent": True, "net_gain": 0.}
        for gain_limit in (0., .01, .03, .06, .12, .20):
            for risk_limit in (.05, .10, .20, .35, 1.):
                act = matched & (values[:, 0] > gain_limit) & (values[:, 1] <= risk_limit)
                benefit = float(np.sum(gain * act / size[group]) / n)
                if benefit <= selected["net_gain"]:
                    continue
                h = float(np.quantile(w @ (harm * act), .95))
                cost = float(np.quantile(w @ (extra * act), .95))
                if h <= limits["policy_risk_delta"] and cost <= limits["policy_mean_extra_tokens_budget"]:
                    decisions[indexes] = act
                    selected = {"gain": gain_limit, "risk": risk_limit, "net_gain": benefit,
                                "observed_risk_upper": h, "extra_tokens_upper": cost}
        by_budget[setting] = selected
    return by_budget, policy_report(rows, decisions)


def config_for(m, row, prediction):
    rule = m["thresholds"][row["setting"]]
    return not rule.get("always_silent", False) and prediction[0] > rule["gain"] and prediction[1] <= rule["risk"]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "refit"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--study", type=Path)
    cli.add_argument("--reference", type=Path)
    args = cli.parse_args()
    torch.set_num_threads(1)
    frozen = read(args.refit / "cohort.json")
    refit = read(args.refit / "gate-model.json")
    labels = args.refit / "paired-labels.json"
    seal = args.refit / "sealed-test.json"
    if (sha_file(labels) != refit["labels_sha256"] or sha_file(seal) != frozen["sealed_test_sha256"]
            or sha_file(args.refit / "cohort.json") != refit["cohort_sha256"]):
        raise ValueError("frozen development input changed")
    base_data = read(labels)
    data = base_data
    features = FEATURES
    if args.study is not None:
        transfer_file = args.study / "transfer-freeze.json"
        if sha_file(transfer_file) != frozen["transfer_sha256"]:
            raise ValueError("archive does not match selected training labels")
        transfer = read(transfer_file)
        manifest = read(Path(transfer["baseline"]) / "manifest.json")
        data = paired_rows(args.study, transfer, manifest, frozen, context_features)
        features = CONTEXT_FEATURES
    development = {split: [r for r in data[split] if r["features"] is not None] for split in data}
    winners, crossval = cross_validate(development["train"], args.study is not None)
    models, summaries = {}, {}
    def fit_thresholds(key, model, val_rows, names):
        matched = [r for r in val_rows if r["features"] is not None]
        vals = predict(matched, model)
        forecast = np.full((len(val_rows), 2), np.nan)
        forecast[[r["features"] is not None for r in val_rows]] = vals
        model["thresholds"], policies = threshold(val_rows, forecast, frozen["limits"])
        model["features"] = names
        models[key] = model
        summaries[key] = {"spec": model["spec"], "shrinkage": model["shrinkage"], "thresholds": model["thresholds"],
                          "prediction": prediction_report(matched, vals, model["baseline"]),
                          "policy": policies}

    for key, (w, loss, l2, shrinkage, length) in winners.items():
        model = fit(development["train"], {"width": w, "loss": loss, "l2": l2, "length_prior": length})
        model["shrinkage"] = shrinkage
        fit_thresholds(key, model, data["validation"], features)
    initial = {"spec": {"width": 8, "loss": "mse", "prior_refit": True},
               "center": refit["center"], "scale": refit["scale"], "absolute": True, "shrinkage": 1.,
               "layers": [[refit["heads"][f"{i}.weight"], refit["heads"][f"{i}.bias"]] for i in (0, 2)],
               "baseline": anchors([r for r in base_data["train"] if r["features"] is not None])}
    fit_thresholds("refit", initial, base_data["validation"], FEATURES)
    if args.reference is not None:
        refit_selection = read(args.reference / "frozen-selection.json")
        previous = read(args.reference / "selection-report.json")
        if (refit_selection["labels_sha256"] != sha_file(labels)
                or refit_selection["seal_sha256"] != sha_file(seal)):
            raise ValueError("incompatible reference training")
        for key, model in refit_selection["models"].items():
            models[f"original-{key}"] = {**model, "features": FEATURES}
            summaries[f"original-{key}"] = previous["validation"][key]
    mlps = [k for k, m in models.items() if m["spec"]["width"] > 0]
    best_width = min(mlps, key=lambda k: (summaries[k]["prediction"]["gain_mae"], models[k]["spec"]["width"]))
    best_policy = max(mlps, key=lambda k: sum(
        v["net_gain"] for v in summaries[k]["thresholds"].values()
    ))
    save(args.out / "frozen-selection.json",
         {"models": models, "selected_mlp": best_width, "selected_gate": best_policy,
          "seal_sha256": sha_file(seal), "labels_sha256": sha_file(labels),
          "transfer_sha256": frozen["transfer_sha256"]})
    save(args.out / "selection-report.json",
         {"selected_mlp": best_width, "selected_gate": best_policy,
          "grouped_cross_validation": crossval, "validation": summaries})
    print("VALIDATION", [(k, v["spec"], v["shrinkage"], v["prediction"]) for k, v in summaries.items()], flush=True)
    print("FROZEN_MLP", best_width, models[best_width]["thresholds"], flush=True)
    print("FROZEN_GATE", best_policy, models[best_policy]["thresholds"], flush=True)


if __name__ == "__main__":
    main()
