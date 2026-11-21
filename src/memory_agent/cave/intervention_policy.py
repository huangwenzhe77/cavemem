"""Fit and deploy a binary intervention policy from pre-decision vectors and real outcomes."""

import math
from statistics import mean

import numpy as np

from . import auxiliary_heads, constrained_gate, gate_mlp
from .logged_policy_fit import pseudo_advantages, risk_upper
from .records import digest
from .replay import require


def fold_id(row):
    return int(digest(row["spec"]["template"])[:12], 16)


def supervised_start(X, gain, harm, base):
    """Supervise the binary policy with measured paired outcomes, keeping the gain regressor separate."""
    initial = [np.asarray(p, dtype=float).copy() for p in base["parameters"]]
    temp = constrained_gate.CONFIG["temperature"]
    cutoff = gate_mlp.CONFIG["threshold"]
    initial[-2] /= temp
    initial[-1] = (initial[-1]-cutoff)/temp
    target = ((gain > cutoff) & (harm == 0)).astype(float)
    for _ in range(200):
        hidden, logits = gate_mlp.forward(X, initial)
        dz = (constrained_gate.sigmoid(logits)-target)/len(X)
        dh = dz[:, None]*initial[2]*(1-hidden**2)
        gradients = (X.T@dh, dh.sum(axis=0), hidden.T@dz, np.array([dz.sum()]))
        for p, g in zip(initial, gradients, strict=True):
            p -= .03*g
    return initial


def logged_scores(rows):
    logs = []
    for row in rows:
        a = row["assignment"]
        arm = a["arm"]
        require(set(row["arms"]) == {arm}, "single-arm row contains an unobserved outcome")
        obs = row["arms"][arm]
        logs.append({"x": row["match"]["e"], "episode_id": row["spec"]["id"],
                     "block": fold_id(row), "z": int(arm == "remind"), "b": a["propensity"],
                     "observed": {"r": obs["utility"], "tokens": obs["tokens"]}})
    for f in {r["block"] % 4 for r in logs}:
        if {r["z"] for r in logs if r["block"] % 4 != f} != {0, 1}:
            return None, []
    return pseudo_advantages(logs, {}, stratum=lambda exposure: exposure)


def train(rows, vectors, config):
    rows = [r for r in rows if r["match"]]
    require(len(set(r["spec"]["template"] for r in rows)) == len(rows), "duplicate training template")
    if len(rows) < 4:
        return {"status": "insufficient_training", "n": len(rows)}
    X = np.stack([vectors(r) for r in rows])
    folds = np.array([fold_id(r) % 4 for r in rows])
    if len(set(folds)) < 2:
        return {"status": "insufficient_task_folds", "n": len(rows)}
    logs = "assignment" in rows[0]
    require(all(("assignment" in r) == logs for r in rows), "mixed estimator cohorts")
    if logs:
        scores, diagnostics = logged_scores(rows)
        if scores is None:
            return {"status": "insufficient_logging_support", "n": len(rows)}
        gain, extra = scores.T
        # The unseen arm permits harm exactly when silent won or remind failed.
        harm = np.array([int(r["arms"]["silent"]["success"] if "silent" in r["arms"]
                            else 1 - r["arms"]["remind"]["success"]) for r in rows])
    else:
        gain = np.array([r["delta"]["utility"] for r in rows])
        extra = np.array([r["delta"]["tokens"] for r in rows])
        harm = np.array([int(r["arms"]["silent"]["success"] > r["arms"]["remind"]["success"])
                         for r in rows])
        diagnostics = []
    base = gate_mlp.train_head(X, gain, np.ones(len(rows)))
    heads = auxiliary_heads.fit_heads(X, extra, harm, folds)
    limits = {"policy_risk_delta": config["risk_budget"],
              "policy_mean_extra_tokens_budget": config["extra_token_budget"]}
    cutoff = gate_mlp.CONFIG["threshold"]
    warm = [np.asarray(p) for p in base["parameters"]]
    if not logs:
        sft = supervised_start(X, gain, harm, base)
        warm = [p.copy() for p in sft]
        # optimize expects a gain-head parameterization before its logit transform.
        warm[-2] *= constrained_gate.CONFIG["temperature"]
        warm[-1] = warm[-1]*constrained_gate.CONFIG["temperature"] + cutoff
    else:
        # Cross-fitted reward targets supply a supervised warm start in the logged setting.
        # No individual rescue or harm label is available.
        sft = [p.copy() for p in warm]
        sft[-2] /= constrained_gate.CONFIG["temperature"]
        sft[-1] = (sft[-1]-cutoff)/constrained_gate.CONFIG["temperature"]
    policy = constrained_gate.optimize(X, gain, harm, extra, warm, cutoff, limits, constrained_gate.CONFIG["kl"])
    return {"status": "fit", "n": len(rows), "supervised": base, "policy": policy, "heads": heads,
            "warm_start": {"parameters": [p.tolist() for p in sft],
                           "objective": "crossfit_gain" if logs else "paired_binary_cross_entropy"},
            "estimator": "logged_DR" if logs else "paired", "crossfit": diagnostics,
            "risk_estimand": "possible_harm_upper_bound" if logs else "paired_regression",
            "limits": limits}


def point(model, x):
    def value(key):
        return float(gate_mlp.forward(x, [np.asarray(p) for p in model[key]["parameters"]])[1])

    heads = auxiliary_heads.predict(model["heads"], x)
    if model["estimator"] == "logged_DR":
        heads["possible_harm_probability"] = heads.pop("regression_probability")
    return {"gain": value("supervised"), "probability": float(constrained_gate.sigmoid(value("policy"))),
            **heads}


def calibrate(model, rows, vectors):
    """Independent task subsets for gain coverage and policy-level harm. Missing evidence abstains."""
    if model["status"] != "fit":
        return {"enabled": False, "reason": model["status"]}
    # Deterministic allocation depends only on immutable template identifiers.
    ordered = sorted(rows, key=fold_id)
    gain_rows = [r for r in ordered[::2] if r["match"]]
    safety_rows = ordered[1::2]
    logged = model["estimator"] == "logged_DR"
    if logged:
        # Individual gain coverage is unidentified in a single-arm cohort.
        # Use a randomized IPS estimate of population gain for the frozen policy.
        samples = []
        for r in ordered[::2]:
            if not r["match"]:
                samples.append(0.)
                continue
            pred = point(model, vectors(r))
            p = pred["probability"] if feasible(pred, model, 0.) else 0.
            a = r["assignment"]
            sign = 1 if a["arm"] == "remind" else -1
            samples.append(sign*p*r["arms"][a["arm"]]["utility"]/a["propensity"])
        if len(samples) < 2:
            return {"enabled": False, "reason": "insufficient_logged_validation"}
        # Task-level approximate lower bound. Do not interpret as per-state gain coverage.
        lower = mean(samples) - 1.96*np.std(samples, ddof=1)/math.sqrt(len(samples))
        offset = None
    else:
        errors = sorted(point(model, vectors(r))["gain"] - r["delta"]["utility"] for r in gain_rows)
        rank = math.ceil(.90*(len(errors)+1))
        if rank > len(errors):
            return {"enabled": False, "reason": "insufficient_gain_calibration", "n": len(errors)}
        offset, lower = float(errors[rank-1]), None
    events, costs = 0, []
    for r in safety_rows:
        if not r["match"]:
            costs.append(0.)
            continue
        pred = point(model, vectors(r))
        allowed = feasible(pred, model, offset or 0.)
        if not allowed:
            costs.append(0.)
            continue
        if logged:
            a = r["assignment"]
            obs = r["arms"][a["arm"]]
            events += int(obs["success"] if a["arm"] == "silent" else 1-obs["success"])
            signed_cost = obs["tokens"]*(1 if a["arm"] == "remind" else -1)/a["propensity"]
        else:
            events += int(r["arms"]["silent"]["success"] > r["arms"]["remind"]["success"])
            signed_cost = r["delta"]["tokens"]
        costs.append(pred["probability"]*signed_cost)
    risk = risk_upper(events, len(safety_rows))
    upper_cost = (float(mean(costs) + 1.96*np.std(costs, ddof=1)/math.sqrt(len(costs)))
                  if len(costs) >= 2 else None)
    enabled = (risk <= model["limits"]["policy_risk_delta"] and upper_cost is not None
               and upper_cost <= model["limits"]["policy_mean_extra_tokens_budget"]
               and (lower is None or lower > 0))
    return {"enabled": bool(enabled), "gain_offset": offset, "logged_policy_gain_lower": lower,
            "reason": "validated" if enabled else "insufficient_validation_evidence",
            "risk_upper": risk, "approximate_extra_token_upper": upper_cost, "n_risk": len(costs)}


def feasible(pred, model, offset):
    limits = model["limits"]
    h = pred.get("regression_probability")
    # Single-arm data can bound population harm but cannot fit an individual H.
    # If the paired head has no event support, held-out population validation still applies.
    return (pred["gain"] > offset and pred["extra_tokens"] <= limits["policy_mean_extra_tokens_budget"]
            and (h is None or h <= limits["policy_risk_delta"]))


def decide(model, x, uniform):
    require(0 <= uniform < 1, "invalid policy draw")
    cal = model["calibration"]
    if not cal["enabled"]:
        return {"remind": False, "probability": 0., "reason": cal["reason"]}
    pred = point(model, x)
    if not feasible(pred, model, cal["gain_offset"] or 0.):
        return {**pred, "probability": 0., "remind": False, "reason": "gain_risk_or_cost_guard"}
    return {**pred, "remind": uniform < pred["probability"], "reason": "frozen_policy"}
