"""Task-fold auxiliary token and regression predictions for the frozen representation."""

import argparse
from pathlib import Path

import numpy as np

from . import formal_plan as fp, gate
from .constrained_gate import Policy, sigmoid, vector
from .records import digest, save
from .replay import require


def logistic(X, y):
    # Ridge logistic head. Leave the intercept unpenalized.
    w = np.zeros(X.shape[1])
    reg = np.eye(len(w))*.5
    reg[-1,-1] = 0.
    for _ in range(35):
        p = sigmoid(X @ w)
        grad = X.T @ (p-y) + reg @ w
        h = (X.T * (p*(1-p))) @ X + reg + np.eye(len(w))*1e-8
        update = np.linalg.solve(h, grad)
        w -= update
        if np.max(np.abs(update)) < 1e-6:
            break
    return w


def fit_heads(X, tokens, harms, folds):
    X = np.column_stack((X, np.ones(len(X))))
    unique = sorted(set(folds))
    require(len(unique) >= 2 and set(harms.tolist()) <= {0,1}, "insufficient independent task folds")
    risk_ok = all(len(set(harms[folds != f].tolist())) == 2 for f in unique)
    ridge = np.eye(X.shape[1])
    ridge[-1,-1] = 0.
    cost_oof = np.zeros(len(X))
    risk_logits = np.zeros(len(X))
    for f in unique:
        held = folds == f
        train = X[~held]
        cost_oof[held] = X[held] @ np.linalg.solve(train.T @ train+ridge, train.T @ tokens[~held])
        if risk_ok:
            risk_logits[held] = X[held] @ logistic(train, harms[~held])
    cost = np.linalg.solve(X.T @ X + ridge, X.T @ tokens)
    risk = None
    if risk_ok:
        calibration = logistic(np.column_stack((risk_logits, np.ones(len(X)))), harms)
        risk = {"weights": logistic(X, harms).tolist(), "platt_oof": calibration.tolist()}
    return {"extra_token_weights": cost.tolist(), "oof_token_mae": float(np.abs(cost_oof-tokens).mean()),
            "risk": risk, "risk_events": int(harms.sum()), "task_count": len(X),
            "risk_status": "fit with out-of-fold logits; external validation required" if risk_ok
            else "insufficient two-class training support across folds"}


def predict(heads, v):
    x = np.append(v, 1.)
    risk = heads["risk"]
    h = None
    if risk:
        a, b = risk["platt_oof"]
        h = float(sigmoid(a*(x @ risk["weights"])+b))
    return {"extra_tokens": float(x @ heads["extra_token_weights"]), "regression_probability": h}


def fit(outputs, model, dest):
    require(not dest.exists(), "new auxiliary release required")
    policy = Policy(model, outputs/"cave-mlp-01/frozen/model.json")
    base, sup = outputs/"cave-formal-01", outputs/"cave-supplement-01"
    rows, _, sources, _ = gate.load_training(base, sup, fp.DEFAULT_BANK)
    require(sources == policy.doc["sources"], "unexpected training cohort")
    costs = {}
    for path in (base/"dataset-v1/train-evidence.jsonl", sup/"deduplicated-training/evidence.jsonl"):
        for e in gate.lines(path):
            arms = {k: e["arms"][k]["observed"] if "arms" in e else e[k] for k in ("remind","silent")}
            costs[e["sample_id"]] = arms["remind"]["total_tokens"]-arms["silent"]["total_tokens"]
    samples = [r["sample"] for r in rows]
    ids = [r["task_id"] for r in rows]
    require(len(set(ids)) == len(ids), "expected a single row per task")
    folds = np.array([int(k[:12],16)%5 for k in ids])
    heads = fit_heads(np.stack([vector(policy.base, s["features"]) for s in samples]),
                      np.array([costs[s["sample_id"]] for s in samples]),
                      np.array([s["targets"]["H"] for s in samples]), folds)
    out = {**heads, "policy_id": policy.doc["policy_id"], "code": fp.code_digest("auxiliary_heads.py"),
           "task_folds_digest": digest(list(zip(ids, folds.tolist()))), "source_digests": sources}
    save(dest, {**out, "id": digest(out)})
    print(dest)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--outputs", type=Path, default=fp.OUTPUT)
    cli.add_argument("--model", required=True, type=Path)
    cli.add_argument("--out", required=True, type=Path)
    opt = cli.parse_args()
    fit(opt.outputs, opt.model, opt.out)
