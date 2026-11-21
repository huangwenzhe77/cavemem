"""Audit a two-stage randomized mini trial; estimate the value of buying a probe."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

from . import formal_plan as fp, gate, policy_value
from .constrained_gate import Policy
from .diagnostic_trial import diagnostic, draw
from .records import digest, read, save
from .replay import require


def train_continuation(train):
    """Two direct action values, two post-inspect action values per observation."""
    direct = [[], []]
    after = [[[], []], [[], []]]
    flags = []
    for r in train:
        if r["first"] == "inspect":
            after[int(r["flag"])][r["z"]].append(r["r"])
            flags.append(int(r["flag"]))
        else:
            direct[r["z"]].append(r["r"])
    if not all(direct) or not flags:
        return None
    mu = [mean(v) for v in direct]
    # Shrink sparse inspection cells to the observed inspected-arm mean.
    # Missing arm support is reported explicitly; never enable a probe from an unsupported fit.
    pooled = [after[0][z]+after[1][z] for z in (0, 1)]
    if not all(pooled):
        return None
    post = [[(sum(v) + 2*mean(pooled[z])) / (len(v)+2) for z, v in enumerate(layer)] for layer in after]
    return {"direct": mu, "post": post, "flag_rate": mean(flags),
            "observed_cells": [[len(v) for v in layer] for layer in after]}


def direct_or_inspect(rows):
    """Out-of-fold assessment of the learned direct and post-inspect decisions."""
    if len(rows) < 3:
        return None
    fitted = {fold: train_continuation([r for r in rows if r["fold"] != fold])
              for fold in {r["fold"] for r in rows}}
    if any(model is None for model in fitted.values()):
        return None
    values = []
    for r in rows:
        m = fitted[r["fold"]]
        a = int(m["direct"][1] > m["direct"][0])
        q = m["direct"][a]
        w = (r["first"] == ("silent", "remind")[a])/r["b1"]
        direct = q + w*(r["r"]-q)
        prob = r["base_p"]
        frozen = (1-prob)*m["direct"][0] + prob*m["direct"][1]
        if r["first"] != "inspect":
            z = r["z"]
            frozen += (prob if z else 1-prob)*(r["r"]-m["direct"][z])/r["b1"]
        maxima = [max(v) for v in m["post"]]
        qi = (1-m["flag_rate"])*maxima[0] + m["flag_rate"]*maxima[1]
        inspect = qi
        wi = 0.
        if r["first"] == "inspect":
            v = m["post"][int(r["flag"])]
            action = int(v[1] > v[0])
            wi = (action == r["z"]) / (r["b1"]*r["b2"])
            inspect += (max(v)-qi)/r["b1"] + wi*(r["r"]-v[r["z"]])
        values.append([direct, inspect, w*r["r"], wi*r["r"], w, wi, frozen])
    return np.array(values)


def estimate(rows):
    # Fixed feasible-to-compute comparison even when folds cannot fit an outcome model.
    # Comparator: direct Silent. Probe continuation: remind iff records would be removed.
    ips = None
    if rows:
        ws = [int(r["first"] == "silent")/r["b1"] for r in rows]
        wi = [(int(bool(r["flag"]) == r["z"])/(r["b1"]*r["b2"])) if r["first"] == "inspect" else 0.
              for r in rows]
        ips = {"gain_vs_direct_silent": mean((a-b)*r["r"] for a,b,r in zip(wi,ws,rows)),
               "matched_direct": sum(w>0 for w in ws), "matched_probe": sum(w>0 for w in wi)}
    values = direct_or_inspect(rows)
    if values is None:
        return {"status": "insufficient fold support", "n": len(rows), "voi_lower": None, "fixed_policy_ips": ips}
    mu = values.mean(axis=0)
    gap = float(mu[1]-mu[0])
    deleted = [direct_or_inspect(rows[:i]+rows[i+1:]) for i in range(len(rows))]
    ci = frozen_ci = None
    if all(v is not None for v in deleted):
        ci = policy_value.jackknife(gap, [float((v[:,1]-v[:,0]).mean()) for v in deleted]).tolist()
        frozen_ci = policy_value.jackknife(float(mu[1]-mu[6]),
                    [float((v[:,1]-v[:,6]).mean()) for v in deleted]).tolist()
    w = values[:,4:6]
    ess = np.divide(w.sum(axis=0)**2, (w*w).sum(axis=0), out=np.zeros(2), where=(w*w).sum(axis=0)>0)
    return {"status": "exploratory", "n": len(rows), "direct_dr": float(mu[0]), "inspect_dr": float(mu[1]),
            "net_voi_dr": gap, "net_voi_ips": float(mu[3]-mu[2]), "voi_ci95_approx": ci,
            "net_voi_vs_frozen": float(mu[1]-mu[6]), "ci95_vs_frozen": frozen_ci,
            "voi_lower": min(ci[0], frozen_ci[0]) if ci else None, "ess_direct_inspect": ess.tolist(),
            "fixed_policy_ips": ips}


def inspection_allowed(evidence, *, uncertain, remaining_probes, risk_validated):
    return (bool(uncertain) and remaining_probes >= 1 and bool(risk_validated) and
            evidence.get("voi_lower") is not None and evidence["voi_lower"] > 0)


def adaptive_decision(features, policy, report, uniform, remaining_probes, risk_validated, inspect):
    """Lazy, single-probe decision. Only a lower bound for uncertain cases may enable it."""
    require(report["policy_id"] == policy.doc["policy_id"], "wrong policy for probe trial")
    budget = features["budget_after_proposal"] if features else {}
    uncertain = bool(features and features["evidence"]["applicable"] == 1 and
                     features["evidence"]["source_verified"] == 1 and features["evidence"]["exposed"] is None and
                     budget["remaining_actor_calls"] >= 1 and budget["remaining_tools"] >= 1)
    model = report["uncertain_continuation"]
    if model and inspection_allowed(report["uncertain"], uncertain=uncertain, remaining_probes=remaining_probes,
                                    risk_validated=risk_validated):
        obs = inspect()
        q = model["post"][int(obs["would_remove_records"])]
        return {"inspected": True, "remind": q[1] > q[0], "observation": obs}
    return {"inspected": False, **policy.decide(features, uniform)}


def audit(root, outputs, dest):
    require(not dest.exists(), "new audit path required")
    reg = gate.signed(root/"preregistration.json", "plan_id")
    policy = Policy(outputs/"cave-primal-dual-01/model.json", outputs/"cave-mlp-01/frozen/model.json")
    plan = gate.signed(outputs/"cave-single-arm-01/preregistration.json", "plan_id")
    seal = read(root/"seal.json")
    require(digest(seal["rows"]) == seal["events_digest"] and reg["policy_id"] == policy.doc["policy_id"]
            and seal["plan_id"] == reg["plan_id"] and plan["plan_id"] == reg["parent_plan"], "invalid seal")
    logged, live, summaries = [], [], []
    for task, entry in zip(reg["tasks"], seal["rows"], strict=True):
        eid, phase = task["episode_id"], task["phase"]
        folder = root/"episodes"/eid
        events = {p.relative_to(folder).as_posix(): read(p) for p in folder.rglob("*.json")}
        require(entry["id"] == eid and entry["events"] == {n: digest(v) for n,v in events.items()}, "raw event changed")
        s = events["summary.json"]
        draws = [n for n in events if n.endswith("/first-stage.json")]
        require(len(draws) <= 1, "more than one decision")
        usage, y = s["usage"], s["verifier"]["reward"]
        r = usage["observed_utility"]
        if r is not None:
            cost = (plan["lambda_tokens"]*usage["total_tokens"]/1000 + plan["lambda_tool"]*usage["tool_calls"]
                    + plan["lambda_time"]*usage["accounted_seconds"]/60)
            execs = [events[k] for k in sorted(events) if k.endswith("/execution.json")]
            submitted = execs[-1]["state"]["submitted"] if execs else None
            expected = sorted(task["task"]["input"], reverse=task["task"]["family"]=="descending")
            require(y == int(submitted == expected) and abs(r-y+cost) < 1e-10, "terminal grade or price")
        summaries.append({"phase": phase, "status": s["main_status"], "y": y, "r": r})
        if not draws:
            continue
        prefix = draws[0].split("/")[0]+"/"
        first, sel = events[draws[0]], events[prefix+"selection.json"]
        x, a = sel["features"], first["first"]
        require(first["selection_digest"] == digest(sel) and sel["pre_digest"] == digest(events[prefix+"pre.json"])
                and sel["draft_digest"] == digest(events[prefix+"draft.json"]), "pre-decision evidence drift")
        u = draw(reg["seed_hex"], eid, "stage1")
        require(first["u1"] == u, "stage-one uniform drift")
        if phase == "binary_live":
            p = policy.probability(x)
            require(first["b1"] == {"silent": 1-p, "remind": p} and a == ("remind" if u < p else "silent"),
                    "online frozen policy mismatch")
        else:
            require(first["b1"] == {a:1/3 for a in reg["stage1"]} and a == reg["stage1"][int(u*3)],
                    "stage-one draw")
        probe = events.get(prefix+"probe.json")
        second = events.get(prefix+"second-stage.json")
        arm, flag, b2 = a, None, None
        if a == "inspect" and second:
            obs = diagnostic(events[prefix+"pre.json"]["state"], x["proposal"])
            u2 = draw(reg["seed_hex"], eid, "stage2")
            require(probe["first_stage_digest"] == digest(first) and probe["observation"] == obs
                    and second["probe_digest"] == digest(probe) and second["u2"] == u2 and second["b2"] == .5
                    and second["arm"] == ("remind" if u2 < .5 else "silent"), "probe or stage-two drift")
            arm, flag, b2 = second["arm"], obs["would_remove_records"], .5
        elif a != "inspect":
            require(not probe and not second, "probe without assignment")
        if r is not None:
            require(a != "inspect" or second, "scored inspection missing second draw")
            exec_tools = sum(len((events[k]["executed_action"] or {}).get("commands", []))
                             for k in events if k.endswith("/execution.json"))
            require(usage["tool_calls"] == exec_tools + int(a == "inspect") and
                    usage["revision_calls"] == int(arm == "remind"), "actual tool or revision costs differ")
            revision = events.get(prefix+"revision-request.json")
            if revision:
                msg = x["history_before_proposal"]
                require(revision["messages"] == [*msg[:-1], {**msg[-1], "content": msg[-1]["content"]
                        +"\n\nAdvisory note for this proposal only: "+x["memory"]["reminder"]}],
                        "probe leaked to actor")
        item = {"episode_id": eid, "fold": task["task"]["template_id"], "first": a, "flag": flag,
                "z": int(arm=="remind") if arm != "inspect" else None, "uncertain": x["evidence"]["exposed"] is None,
                "r": r, "y": y, "b1": first["b1"][a], "b2": b2, "base_p": policy.probability(x)}
        (live if phase == "binary_live" else logged).append(item)
    complete = all(s["r"] is not None for s in summaries if s["phase"] == "inspect_explore")
    uncertain = [r for r in logged if r["uncertain"]]
    eligible = estimate(logged) if complete else {"status": "missing registered outcome", "voi_lower": None}
    subset = estimate(uncertain) if complete else {"status": "missing registered outcome", "voi_lower": None}
    model = train_continuation(uncertain) if complete else None
    report = {"plan_id": reg["plan_id"], "policy_id": reg["policy_id"], "seal_digest": digest(seal),
              "code": fp.code_digest("probe_value.py"), "eligible": eligible, "uncertain": subset,
              "uncertain_continuation": model, "logged": logged, "binary_decisions": live,
              "funnel": {p: dict(Counter(s["status"] for s in summaries if s["phase"] == p))
                         for p in ("inspect_explore","binary_live")},
              "live_successes": sum(s["y"] == 1 for s in summaries if s["phase"] == "binary_live"),
              "missing_utility": sum(s["r"] is None for s in summaries),
              "first_counts": dict(Counter(r["first"] for r in logged)),
              "inspection_recommended": inspection_allowed(subset, uncertain=True, remaining_probes=1,
                                                          risk_validated=False)}
    save(dest, report)
    print(dest)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--outputs", type=Path, default=fp.OUTPUT)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    audit(args.root, args.outputs, args.out)
