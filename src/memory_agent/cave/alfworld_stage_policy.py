"""Learn separate rescue, regression and token forecasts from single-reminder pairs."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from memory_agent.cave.alfworld_env import sha_file
from memory_agent.cave.alfworld_gate_refit import policy_report
from memory_agent.cave.alfworld_gate_tune import SETTINGS, normalize, weights
from memory_agent.cave.alfworld_head_pilot import FEATURES
from memory_agent.cave.records import read, save


SEED = 262737
STEPS = 300
RISK_LIMIT = .05


def baseline(rows):
    estimates = {}
    for setting in SETTINGS:
        part = [r for r in rows if r["setting"] == setting]
        rescue = (sum(r["R"] for r in part) + .5) / (len(part) + 1.5)
        harm = (sum(r["H"] for r in part) + .5) / (len(part) + 1.5)
        extra = float(np.mean([r["extra"] for r in part])) / 1000
        estimates[setting] = (rescue, harm, extra)
    return estimates


def fit(rows, width, penalty):
    base = baseline(rows)
    mass = weights(rows)
    x0 = np.asarray([r["features"] for r in rows])
    center = mass @ x0
    scale = np.maximum(np.sqrt(mass @ (x0 - center)**2), .01)
    x = torch.tensor(normalize(rows, center, scale), dtype=torch.float32)
    obs = torch.tensor([1 if r["R"] else 2 if r["H"] else 0 for r in rows])
    cost = torch.tensor([r["extra"] / 1000 for r in rows], dtype=x.dtype)
    origin = torch.tensor([base[r["setting"]] for r in rows], dtype=x.dtype)
    log_baseline = torch.log(torch.column_stack((1 - origin[:, 0] - origin[:, 1], origin[:, :2])))
    mass = torch.tensor(mass, dtype=x.dtype)
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(x.shape[1], width), nn.Tanh(), nn.Linear(width, 3)) if width else nn.Linear(x.shape[1], 3)
    opt = torch.optim.Adam(model.parameters(), lr=.008)
    for _ in range(STEPS):
        opt.zero_grad()
        residual = model(x)
        # Use proper multiclass likelihood for mutually exclusive rescue and harm.
        logits = log_baseline + torch.column_stack((torch.zeros(len(x)), residual[:, :2]))
        classification = F.cross_entropy(logits, obs, reduction="none")
        price = F.huber_loss(residual[:, 2] + origin[:, 2], cost, reduction="none", delta=1.)
        objective = mass @ (classification + .25 * price) + penalty * sum(p.square().sum() for p in model.parameters())
        objective.backward()
        opt.step()
    layers = (model[0], model[2]) if width else (model,)
    return {"width": width, "penalty": penalty, "steps": STEPS, "center": center.tolist(),
            "scale": scale.tolist(), "baseline": base,
            "layers": [[layer.weight.detach().tolist(), layer.bias.detach().tolist()] for layer in layers]}


def predict(rows, model):
    result = np.zeros((len(rows), 3))
    take = [i for i, r in enumerate(rows) if r["features"] is not None]
    if not take:
        return result
    part = [rows[i] for i in take]
    x = normalize(part, model["center"], model["scale"])
    for i, (kernel, bias) in enumerate(model["layers"]):
        if i:
            x = np.tanh(x)
        x = x @ np.asarray(kernel).T + bias
    origin = np.asarray([model["baseline"][r["setting"]] for r in part])
    logits = np.log(np.column_stack((1 - origin[:, :2].sum(axis=1), origin[:, :2])))
    logits[:, 1:] += x[:, :2]
    mass = np.exp(logits - logits.max(axis=1, keepdims=True))
    result[take, :2] = (mass / mass.sum(axis=1, keepdims=True))[:, 1:]
    result[take, 2] = 1000 * (origin[:, 2] + x[:, 2])
    return result


def grouped_brier(rows, forecast):
    mass = weights(rows)
    rescue = np.asarray([r["R"] for r in rows])
    harm = np.asarray([r["H"] for r in rows])
    extra = np.asarray([r["extra"] for r in rows])
    return {"rescue_brier": float(mass @ (rescue - forecast[:, 0])**2),
            "harm_brier": float(mass @ (harm - forecast[:, 1])**2),
            "token_mae": float(mass @ abs(extra - forecast[:, 2]))}


def cross_validate(train):
    templates = sorted({r["template"] for r in train})
    np.random.default_rng(SEED).shuffle(templates)
    by_template = {key: i % 4 for i, key in enumerate(templates)}
    groups = np.array([by_template[r["template"]] for r in train])
    reports, candidates = {}, []
    for width in (0, 8, 16):
        for penalty in (.003, .03, .10):
            values = np.zeros((len(train), 3))
            for fold in range(4):
                learn = [r for r, g in zip(train, groups, strict=True) if g != fold]
                hidden = [r for r, g in zip(train, groups, strict=True) if g == fold]
                values[groups == fold] = predict(hidden, fit(learn, width, penalty))
            results = grouped_brier(train, values)
            key = f"{width}/{penalty}"
            reports[key] = results
            value = results["rescue_brier"] + results["harm_brier"] + .01 * results["token_mae"] / 1000
            candidates.append((value, width, penalty))
    top = min(v for v in candidates if v[1] > 0)
    top_linear = min(v for v in candidates if v[1] == 0)
    return top[1:], top_linear[1:], reports


def decide(rows, forecast, policies):
    answer = []
    for row, (r, h, tokens) in zip(rows, forecast, strict=True):
        gate = policies[row["setting"]]
        if gate.get("always_silent") or row["features"] is None:
            answer.append(False)
            continue
        score = r - gate["harm_weight"] * h - gate["token_weight"] * tokens / 1000
        answer.append(bool(score > gate["threshold"]))
    return answer


def select(rows, forecast):
    rules = {}
    for setting in SETTINGS:
        indexes = [i for i, row in enumerate(rows) if row["setting"] == setting]
        part = [rows[i] for i in indexes]
        m = np.asarray([row["features"] is not None for row in part])
        r, h, predicted_tokens = forecast[indexes].T
        win = np.asarray([row["R"] - row["H"] for row in part])
        harm = np.asarray([row["H"] for row in part])
        tokens = np.asarray([row["extra"] for row in part])
        names = [row["template"] for row in part]
        _, ids, size = np.unique(names, return_inverse=True, return_counts=True)
        groups = len(size)
        draws = np.random.default_rng(SEED).multinomial(groups, np.full(groups, 1/groups), size=1024) / groups
        bootstrap = draws[:, ids] / size[ids]
        rules[setting] = {"always_silent": True}
        best = (0, 0, 0, 0)
        for harm_weight in (.5, 1., 2.):
            for token_weight in (0., .02, .08, .20):
                scores = r - harm_weight * h - token_weight * predicted_tokens / 1000
                for threshold in (0., .015, .035, .075, .15):
                    take = m & (scores > threshold)
                    wins, extra, harms = int(take @ win), int(take @ tokens), int(take @ harm)
                    # Rank improvements that dominate both controls first.
                    dominates = wins > max(0, int(win.sum())) and extra < min(0, int(tokens.sum()))
                    key = int(dominates), wins, -extra, -harms
                    if key <= best or np.quantile(bootstrap @ (harm * take), .95) > RISK_LIMIT:
                        continue
                    if not dominates and (wins < 0 or extra > 0):
                        continue
                    rules[setting] = {"harm_weight": harm_weight, "token_weight": token_weight,
                                      "threshold": threshold, "development_dominates": dominates}
                    best = key
    return rules


def review(rows, model, comparator, agreement_index=0):
    forecast = predict(rows, model)
    forecasts_linear = predict(rows, comparator)
    rules = select(rows, forecast)
    rules_linear = select(rows, forecasts_linear)
    actions = {
        "silent": [False] * len(rows), "always_remind": [r["features"] is not None for r in rows],
        "mlp": decide(rows, forecast, rules), "linear": decide(rows, forecasts_linear, rules_linear),
        "action_disagrees": [r["features"] is not None and r["features"][agreement_index] == 0 for r in rows],
        "clairvoyant_ceiling": [(r["R"] - r["H"], -r["extra"]) > (0, 0) for r in rows],
    }
    eligible = [i for i, r in enumerate(rows) if r["features"] is not None]
    part = [rows[i] for i in eligible]
    metrics = {"mlp": grouped_brier(part, forecast[eligible]),
               "linear": grouped_brier(part, forecasts_linear[eligible]),
               "budget_prior": grouped_brier(part, np.array([
                   [*model["baseline"][r["setting"]][:2], 1000 * model["baseline"][r["setting"]][2]]
                   for r in part
               ]))}
    return rules, rules_linear, {k: policy_report(rows, v) for k, v in actions.items()}, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = args.out / "paired-labels.json"
    dataset = read(path)
    train = [r for r in dataset["train"] if r["features"] is not None]
    valid = dataset["validation"]
    if {r["template"] for r in train} & {r["template"] for r in valid}:
        raise ValueError("development split shares templates")
    categories = Counter("rescue" if r["R"] else "harm" if r["H"] else "other" for r in train)
    if not categories["rescue"] or not categories["harm"]:
        raise ValueError("the candidate intervention did not produce both outcome types in training")
    mlp_spec, linear_spec, crossval = cross_validate(train)
    model, linear = fit(train, *mlp_spec), fit(train, *linear_spec)
    legacy = read(args.out / "collector-freeze.json")["selection"] == "legacy-wide-persistent"
    rules, rules_linear, policies, metrics = review(valid, model, linear, FEATURES.index("original_agrees") if legacy else 0)
    save(args.out / "model.json", {"model": model, "linear": linear, "rules": rules, "linear_rules": rules_linear,
                                   "dataset_sha256": sha_file(path),
                                   "collector_sha256": sha_file(args.out / "collector-freeze.json")})
    save(args.out / "development-report.json", {"training_labels": dict(categories), "cross_validation": crossval,
                                                "validation_prediction": metrics, "validation_policies": policies})
    print("STAGE_POLICY_FIT", dict(categories), mlp_spec, metrics, rules, flush=True)


if __name__ == "__main__":
    main()
