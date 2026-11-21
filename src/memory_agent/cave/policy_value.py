"""Card 10. Fit a threshold policy on paired training data and assess frozen policies."""

import argparse
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from . import formal, formal_plan as fp, gate, gate_mlp
from .logged_data import load_logged_dataset
from .records import digest, read, save
from .replay import require


POLICIES = ("silent", "always_remind", "ready_rule", "mlp", "constrained_threshold")
METRICS = ("utility", "success", "tokens")


def choose(score, threshold):
    return threshold is not None and score > threshold


def fit_threshold(outputs, runtime, plan):
    base, sup = outputs / "cave-formal-01", outputs / "cave-supplement-01"
    rows, cards, sources, _ = gate.load_training(base, sup, fp.DEFAULT_BANK)
    model = runtime.model
    require(cards == model["cards"] and sources == model["sources"] and
            model["bank_id"] == plan["bank_id"] and model["label_spec"] == plan["label_spec"],
            "paired training and frozen policy incompatible")
    evidence = {}
    for i, path in enumerate((base / "dataset-v1/train-evidence.jsonl",
                              sup / "deduplicated-training/evidence.jsonl")):
        for e in gate.lines(path):
            arms = e if i else {a: e["arms"][a]["observed"] for a in ("remind", "silent")}
            evidence[e["sample_id"]] = arms["remind"]["total_tokens"] - arms["silent"]["total_tokens"]
    targets = []
    for row in rows:
        s = row["sample"]
        targets.append((runtime.predict(s["features"])["predicted_D"], s["targets"]["D"],
                        s["targets"]["H"], evidence[s["sample_id"]]))
    require(len(set(r["task_id"] for r in rows)) == len(rows), "expected one pair per training task")
    spec = plan["label_spec"]
    candidates = []
    # Fixed score grid plus a silence fallback. The encoder and head stay frozen.
    for threshold in (None, model["threshold"], .05, .1, .25, .5, 1.):
        actions = [choose(t[0], threshold) for t in targets]
        gains, risks, costs = (mean(a * t[j] for a, t in zip(actions, targets)) for j in (1, 2, 3))
        candidates.append({"threshold": threshold, "reminders": sum(actions), "gain": gains,
                           "observed_risk": risks, "extra_tokens": costs, "empirically_feasible":
                           risks <= spec["policy_risk_delta"] and costs <= spec["policy_mean_extra_tokens_budget"]})
    best = max((c for c in candidates if c["empirically_feasible"]),
               key=lambda c: (c["gain"], -c["reminders"]))
    return {"candidates": candidates, "selected": best, "n": len(rows), "risk_events": sum(t[2] for t in targets),
            "sources": sources, "model_id": model["model_id"], "sample": "paired training only"}


def predictions(rows, runtime, threshold):
    out = []
    for row in rows:
        x = row["x"]
        funcs = (lambda: False, lambda: True, lambda: not gate.ready(x, runtime.model["cards"]),
                 lambda: runtime.predict(x)["remind"],
                 lambda: threshold is not None and choose(runtime.predict(x)["predicted_D"], threshold))
        actions, seconds = {}, {}
        for policy, fn in zip(POLICIES, funcs):
            actions[policy] = int(fn())
            start = perf_counter()
            for _ in range(51):
                fn()
            seconds[policy] = (perf_counter() - start) / 51
        out.append({"decision_id": row["decision_id"], "cluster": row["cluster_id"], "fold": row["template_id"],
                    "ready": not actions["ready_rule"], "actions": actions, "policy_seconds": seconds})
    return out


def dr_scores(rows, pred, time_price, cap=2.):
    """Leave out the entire wording. In each arm shrink readiness cells to that arm's training mean."""
    outcomes = np.array([[r["r"], r["y"], r["usage"]["total_tokens"]] for r in rows])
    scores, weights, folds = [], [], {}
    for i, (r, target) in enumerate(zip(rows, pred, strict=True)):
        fold = target["fold"]
        if fold not in folds:
            train = [j for j, p in enumerate(pred) if p["fold"] != fold]
            folds[fold] = [[j for j in train if rows[j]["z"] == arm] for arm in (0, 1)]
        mu = []
        for indices in folds[fold]:
            require(indices, "no training support for arm in held-out wording")
            same = [j for j in indices if pred[j]["ready"] == target["ready"]]
            prior = outcomes[indices].mean(axis=0)
            mu.append((2 * prior + outcomes[same].sum(axis=0)) / (2 + len(same)))
        z = r["z"]
        values, importance = [], []
        for policy in POLICIES:
            a = target["actions"][policy]
            w = min(cap, 1 / r["p_logged"]) if a == z else 0.
            value = mu[a] + w * (outcomes[i] - mu[z])
            value[0] -= time_price * target["policy_seconds"][policy]
            values.append(value)
            importance.append(w)
        scores.append(values)
        weights.append(importance)
    return np.array(scores), np.array(weights)


def jackknife(full, deleted):
    # Delete one independent task, including refitting the nuisance regressions.
    vals = np.asarray(deleted)
    se = np.sqrt((len(vals) - 1) / len(vals) * ((vals - vals.mean(axis=0)) ** 2).sum(axis=0))
    return np.stack((full - 1.96 * se, full + 1.96 * se), axis=-1)


def evaluate(rows, episodes, pred, time_price, limits):
    require(len(rows) > 2 and len({e["cluster_id"] for e in episodes}) == len(episodes)
            and all(e["r"] is not None and e["y"] is not None for e in episodes),
            "missing outcome or non-independent task; no complete-case evaluation")
    n, total = len(rows), len(episodes)
    scores, w = dr_scores(rows, pred, time_price)
    base = scores.mean(axis=0)
    delta = base - base[0]
    # At unassigned tasks every target policy has the same observed trajectory.
    unassigned = [e for e in episodes if e["assigned_arm"] is None]
    common = np.array([[e["r"], e["y"], e["usage"]["total_tokens"]] for e in unassigned]).reshape(-1, 3)
    marginal = (scores.sum(axis=0) + common.sum(axis=0)) / total
    deletes, all_deletes = [], []
    for i in range(n):
        vals, _ = dr_scores(rows[:i] + rows[i+1:], pred[:i] + pred[i+1:], time_price)
        m = vals.mean(axis=0)
        deletes.append(m - m[0])
        all_deletes.append((m - m[0]) * (n-1) / (total-1))
    all_deletes.extend([delta * n / (total-1)] * len(common))
    conditional_ci = jackknife(delta, deletes)
    all_ci = jackknife(delta * n / total, all_deletes)
    vs_rule = jackknife(delta - delta[2], [v - v[2] for v in deletes])
    clipped = dr_scores(rows, pred, time_price, cap=1.)[0].mean(axis=0)
    reports = {}
    for j, policy in enumerate(POLICIES):
        support = float(w[:,j].sum())
        # For a reminded target, H may be 1 only if the observed arm does not rule it out.
        possible_harms = sum(p["actions"][policy] and (r["z"] != r["y"])
                             for r, p in zip(rows, pred, strict=True))
        reports[policy] = {
            "scheduled_reminders": sum(p["actions"][policy] for p in pred),
            "ess": support**2 / float((w[:,j]**2).sum()) if support else 0.,
            "eligible_value": dict(zip(METRICS, base[j].tolist())),
            "all_task_value": dict(zip(METRICS, marginal[j].tolist())),
            "eligible_delta_vs_silent": dict(zip(METRICS, delta[j].tolist())),
            "eligible_delta_ci95": dict(zip(METRICS, conditional_ci[j].tolist())),
            "all_delta_vs_silent": dict(zip(METRICS, (delta[j]*n/total).tolist())),
            "all_delta_ci95": dict(zip(METRICS, all_ci[j].tolist())),
            "eligible_gain_vs_rule": float(delta[j,0] - delta[2,0]),
            "eligible_gain_vs_rule_ci95": vs_rule[j,0].tolist(),
            "eligible_gain_clip1_vs_silent": float(clipped[j,0] - clipped[0,0]),
            "overlap": {"logging_probability": .5, "max_weight": float(w[:,j].max()),
                        "matching_tasks": int((w[:,j] > 0).sum())},
            "constraints": {"all_task_token_upper_ci_within_budget":
                            bool(all_ci[j,2,1] <= limits["policy_mean_extra_tokens_budget"]),
                            "observed_cohort_possible_regressions": [0, int(possible_harms)],
                            "cohort_upper_fraction": possible_harms / total,
                            "population_risk_status": "zero by construction" if policy == "silent"
                            else "not certified"},
        }
    return reports


def paired_validation(folder, model, selected, time_price):
    ledger = gate.lines(folder / "policy-episodes.jsonl")
    summary = read(folder / "comparison.json")
    pp = gate.signed(folder / "predictions.json", "id")
    require(digest(ledger) == summary["ledger_digest"] and pp["model_digest"] == digest(model)
            and pp["plan_id"] == summary["plan_id"], "paired validation lineage")
    totals = {p: Counter() for p in POLICIES}
    for e, pre in zip(ledger, pp["rows"], strict=True):
        require(e["episode_id"] == pre["episode_id"], "unaligned paired validation")
        known = e["policy"]
        for p in POLICIES:
            if p != "constrained_threshold":
                pick = known[p]
                utility = pick["utility"]
            else:
                preds = pre["predictions"]
                intervene = bool(preds) and choose(preds["mlp"]["predicted_D"], selected)
                arm = "always_remind" if intervene else "silent"
                pick = known[arm]
                t = pre["gate_seconds"]
                # Use the same frozen head's recorded runtime when evaluating its alternative threshold.
                extra_time = (t["mlp"] - t[arm]) if preds and selected is not None else 0.
                utility = pick["utility"] - time_price * extra_time
            totals[p].update({"success": pick["success"], "reminders": int(pick["remind"]),
                              "regressions": int(known["silent"]["success"] == 1 and pick["success"] == 0),
                              "tokens": pick["total_tokens"], "utility": utility})
    return {"tasks": len(ledger), "paired": summary["pairs"], "totals": totals}


def run(outputs, dest):
    require(not dest.exists(), "refusing to overwrite evaluation")
    root = outputs / "cave-single-arm-01"
    rows, episodes, ds = load_logged_dataset(root / "dataset-v1")
    plan = gate.signed(root / "preregistration.json", "plan_id")
    require(ds["plan_id"] == plan["plan_id"] and plan["bank_id"] == ds["bank_id"], "wrong plan")
    model_dir = outputs / "cave-mlp-01"
    runtime = gate_mlp.load_model(model_dir / "frozen/model.json")
    require(set(runtime.model["task_ids"]).isdisjoint(e["cluster_id"] for e in episodes), "train overlap")
    fit = fit_threshold(outputs, runtime, plan)
    threshold = fit["selected"]["threshold"]
    pred = predictions(rows, runtime, threshold)
    price = plan["lambda_time"] / 60
    report = {"dataset_id": ds["dataset_id"], "plan_id": ds["plan_id"], "model_id": runtime.model["model_id"],
              "cold_model_load_seconds": runtime.load_seconds, "fit_digest": digest(fit),
              "prediction_digest": digest(pred), "code": fp.code_digest("policy_value.py"),
              "eligible": len(rows), "tasks": len(episodes), "limits": plan["label_spec"],
              "fold_counts": {f: dict(Counter(r["arm"] for r in rows if r["template_id"] == f))
                              for f in plan["online_templates"]},
              "online": evaluate(rows, episodes, pred, price, plan["label_spec"]),
              "paired": {s: paired_validation(model_dir / s, runtime.model, threshold, price)
                         for s in ("formal", "previous", "prospective")},
              "risk_status": "individual counterfactual harm not identified in single-arm records"}
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp) / "release"
        stage.mkdir()
        save(stage / "training.json", fit)
        formal.write_rows(stage / "decisions.jsonl", pred)
        save(stage / "evaluation.json", report)
        stage.rename(dest)
    print(dest / "evaluation.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=fp.OUTPUT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.outputs, args.out)
