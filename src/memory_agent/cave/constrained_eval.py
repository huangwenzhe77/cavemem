"""Evaluate a frozen stochastic policy on disjoint paired and randomized development sets."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from . import formal_plan as fp, gate, policy_value as pv
from .constrained_gate import Policy
from .logged_data import load_logged_dataset
from .records import digest, read, save
from .replay import require


NAMES = ("silent", "always_remind", "ready_rule", "supervised", "kl", "no_kl")
METRICS = ("utility", "success", "total_tokens")


def decisions(policy, x):
    funcs = (lambda: 0., lambda: 1., lambda: not gate.ready(x, policy.base.model["cards"]),
             lambda: policy.base.predict(x)["remind"], lambda: policy.probability(x, "kl"),
             lambda: policy.probability(x, "no_kl"))
    probs, seconds = [], []
    for fn in funcs:
        probs.append(float(fn()))
        start = perf_counter()
        for _ in range(21):
            fn()
        seconds.append((perf_counter()-start)/21)
    return np.array(probs), np.array(seconds)


def paired(outputs, cohort, archive, policy, price):
    root = outputs / cohort / "dataset-v1"
    rel = gate.signed(root / "dataset.json", "dataset_id")
    samples = gate.lines(root / "validation.jsonl")
    require(digest(samples) == rel["splits"]["validation"]["samples_digest"], "paired feature digest")
    xs = {s["sample_id"]: s["features"] for s in samples}
    old = outputs / "cave-mlp-01" / archive
    entries = gate.lines(old / "policy-episodes.jsonl")
    comp = read(old / "comparison.json")
    pred = gate.signed(old / "predictions.json", "id")
    require(digest(entries) == comp["ledger_digest"] and comp["plan_id"] == pred["plan_id"] == rel["plan_id"]
            and pred["model_digest"] == digest(policy.base.model) and
            set(xs) == {e["sample_id"] for e in entries if e["sample_id"]}, "validation lineage")
    metrics = np.zeros((len(NAMES), 5))
    for e, logged in zip(entries, pred["rows"], strict=True):
        require(e["episode_id"] == logged["episode_id"], "task order")
        t = logged["gate_seconds"]
        arms = []
        for name in ("silent", "always_remind"):
            obs = e["policy"][name]
            arms.append(np.array([obs[k] for k in METRICS]))
            arms[-1][0] += price*t.get(name, 0.)
        key = e["sample_id"]
        p = sec = np.zeros(len(NAMES))
        if key:
            require(digest(xs[key]) == logged["features_digest"], "decision feature lineage")
            p, sec = decisions(policy, xs[key])
        m = arms[0] + p[:, None]*(arms[1]-arms[0])
        m[:,0] -= price*sec
        metrics += np.column_stack((m, p, p * int(e["cell"] == "regression")))
    return {"registered": len(entries), "pairs": len(xs), "estimand": "expected counts under policy randomization",
            "totals": {k: dict(zip((*METRICS, "reminders", "regressions"), r.tolist()))
                       for k, r in zip(NAMES, metrics)}}


def online(outputs, policy, price):
    rows, episodes, release = load_logged_dataset(outputs / "cave-single-arm-01/dataset-v1")
    require(release["bank_id"] == policy.doc["bank_id"] and
            set(policy.base.model["task_ids"]).isdisjoint(e["cluster_id"] for e in episodes)
            and all(e["r"] is not None for e in episodes), "invalid online split")
    probs, secs, nuisance = [], [], []
    for row in rows:
        p, t = decisions(policy, row["x"])
        probs.append(p)
        secs.append(t)
        nuisance.append({"fold": row["template_id"], "ready": not p[2],
                         "actions": {k: int(k == "always_remind") for k in pv.POLICIES},
                         "policy_seconds": dict.fromkeys(pv.POLICIES, 0.)})
    probs, secs = np.array(probs), np.array(secs)
    def scores(indices):
        scores_by_arm, _ = pv.dr_scores([rows[i] for i in indices], [nuisance[i] for i in indices], 0.)
        p = probs[indices]
        # DR is linear in the target action probabilities; charged inference is deterministic.
        result = scores_by_arm[:,0,None,:] + p[:,:,None]*(scores_by_arm[:,1,None,:]-scores_by_arm[:,0,None,:])
        result[:,:,0] -= price*secs[indices]
        return result.sum(axis=0)
    n, total = len(rows), len(episodes)
    indices = list(range(n))
    s = scores(indices)
    diff = (s-s[0])/total
    deleted = []
    for i in indices:
        other = scores(indices[:i]+indices[i+1:])
        deleted.append((other-other[0])/(total-1))
    deleted += [(s-s[0])/(total-1)]*(total-n)
    ci = pv.jackknife(diff, deleted)
    z = np.array([r["z"] for r in rows])[:,None]
    w = np.where(z, probs, 1-probs)/np.array([r["p_logged"] for r in rows])[:,None]
    return {"dataset_id": release["dataset_id"], "registered": total, "assigned": n,
            "policies": {k: {"all_gain_vs_silent": float(diff[j,0]), "ci95_approx": ci[j,0].tolist(),
                             "all_extra_tokens": float(diff[j,2]), "expected_reminders": float(probs[:,j].sum()),
                             "ess": float(w[:,j].sum()**2 / (w[:,j]**2).sum())} for j,k in enumerate(NAMES)}}


def evaluate(outputs, model, dest):
    require(not dest.exists(), "use a new evaluation path")
    policy = Policy(model, outputs / "cave-mlp-01/frozen/model.json")
    price = gate.signed(outputs / "cave-formal-01/preregistration.json", "plan_id")["lambda_time"]/60
    roots = (("formal", "cave-formal-01"), ("previous", "cave-gate-validation-01"),
             ("prospective", "cave-mlp-validation-01"))
    report = {"policy_id": policy.doc["policy_id"], "code": fp.code_digest("constrained_eval.py"),
              "paired": {s: paired(outputs, r, s, policy, price) for s,r in roots},
              "logged": online(outputs, policy, price)}
    save(dest, report)
    print(dest)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--outputs", type=Path, default=fp.OUTPUT)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    evaluate(args.outputs, args.model, args.out)
