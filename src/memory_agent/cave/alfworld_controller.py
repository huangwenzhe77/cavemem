"""Run the CAVE controller at an ALFWorld checkpoint before executing a command."""

import random
import re
from time import monotonic

import numpy as np

from . import gate_mlp, intervention_policy, probe_value
from .alfworld import MAX_STEPS, PRICES, roll, snapshot
from .alfworld_env import Episode, sha_file
from .alfworld_memory import matched
from .logged_policy_fit import risk_upper
from .records import digest, read, save
from .replay import require


def uniform(spec, stage, step):
    return random.Random(digest([spec["seed"], spec["id"], stage, step])).random()


def pattern(action):
    return re.sub(r" \d+(?= |$)", " #", action)


def load_bank(root, cards):
    enriched = []
    for card in cards:
        receipt = (root / card["evidence"]).resolve()
        receipt.relative_to(root.resolve())
        require(sha_file(receipt) == card["evidence_sha256"], "source evidence digest mismatch")
        evidence = read(receipt)
        require(evidence["admitted"] and evidence["source"] == card["source"] and
                all(card[k] == v for k,v in evidence["card"].items()), "invalid source evidence")
        cp = evidence["checkpoint"]
        enriched.append({**card, "source_original": cp["draft"]["action"],
                         "source_checked": cp["context"]["checked"],
                         "source_menu": cp["draft"]["allowed"]})
    return enriched


def assess(cp, candidate):
    card, a0 = candidate["card"], cp["draft"]["action"]
    e = (0 if a0 in candidate["recommended"] else
         1 if pattern(a0) == pattern(card["source_original"]) else None)
    before = {pattern(c) for c in card["source_menu"]}
    current = {pattern(c) for c in cp["draft"]["allowed"]}
    divergence = 1-len(before & current)/max(1, len(before | current))
    return {**candidate, "v": 1, "rho": 1, "e": e, "exposure_known": e is not None,
            "exposure_basis": {0: "matches_recommendation", 1: "matches_source_failure",
                               None: "no_matching_signature"}[e],
            "menu_divergence": divergence,
            "visit_count_difference": len(cp["context"]["checked"])-len(card["source_checked"]),
            "remaining_steps": MAX_STEPS - cp["step"] + 1}


def vector(row, tokenizer, table):
    from .alfworld_experiment import PHASES

    cp, m = row["checkpoint"], row["match"]
    ctx, card = cp["context"], m["card"]
    text = "\n".join((ctx["goal"], ctx["observation"], card["mechanism"], card["reminder"]))
    quality = card.get("q", {}).get("remind", {})
    interval = quality.get("ci95") or [0., 0.]
    return np.concatenate((gate_mlp.pool(text, gate_mlp.CONFIG["context_tokens"], tokenizer, table),
                           gate_mlp.pool(cp["draft"]["action"], gate_mlp.CONFIG["action_tokens"], tokenizer, table),
                           [ctx["phase"] == p for p in PHASES], [m["v"], m["rho"], m["e"] == 1,
                            m["exposure_known"], m["menu_divergence"], m["visit_count_difference"]/MAX_STEPS,
                            m["remaining_steps"]/MAX_STEPS, np.log1p(quality.get("k", 1)),
                            quality.get("mean", 0.), interval[0], interval[1]-interval[0]]))


def inspect_prefix(prefix, action):
    """Read-only check for a previously unproductive attempt at this exact command."""
    return any(r["execution"]["action"] == action and
               r["context"]["observation"] == r["execution"]["observation"] for r in prefix)


def decision_context(row):
    card = row["match"]["card"]
    return digest([row["spec"]["family"], card["phase"], pattern(card["source_original"]), card["pattern"]])


def in_domain(model, row):
    return ("validated_contexts" not in model or decision_context(row) in model["validated_contexts"])


def select(row, mode, runtime, prefix):
    spec, cp = row["spec"], row["checkpoint"]
    draw = uniform(spec, "arm", cp["step"])
    if mode == "single_arm":
        return {"arm": "remind" if draw < .5 else "silent", "propensity": .5, "draw": draw}
    if mode == "diagnostic":
        model, encoder = runtime
        x = vector(row, *encoder) if encoder else None
        base_p = intervention_policy.decide(model, x, draw)["probability"] if in_domain(model, row) else 0.
        first = ("silent", "remind", "inspect")[int(3*draw)]
        flag = None
        arm, b2 = first, 1.
        if first == "inspect":
            flag = inspect_prefix(prefix, cp["draft"]["action"])
            arm = "remind" if uniform(spec, "after_inspect", cp["step"]) < .5 else "silent"
            b2 = .5
        return {"first": first, "arm": arm, "flag": flag, "b1": 1/3, "b2": b2, "base_p": base_p}
    model, encoder = runtime
    if not in_domain(model, row):
        return {"arm": "silent", "remind": False, "probability": 0., "reason": "outside_validation_support",
                "inspected": False}
    x = vector(row, *encoder) if encoder else None
    decision = intervention_policy.decide(model, x, draw)
    report = model.get("probe", {})
    inspected = probe_value.inspection_allowed(report, uncertain=row["match"]["e"] is None,
                                               remaining_probes=1, risk_validated=report.get("validated", False))
    if inspected:
        flag = inspect_prefix(prefix, cp["draft"]["action"])
        values = report["continuation"]["post"][int(flag)]
        decision = {"remind": values[1] > values[0], "reason": "validated_inspect", "flag": flag}
    return {**decision, "arm": "remind" if decision["remind"] else "silent", "inspected": inspected}


def collect_one(args, spec, actor, bank, protocol, mode, dest, runtime=None):
    from .alfworld_experiment import choose_card, costs

    prefix, checks = [], []
    survival = 1.
    overhead = 0.
    ep = Episode(args.data, args.config, spec)
    try:
        while not ep.terminal:
            draft = actor.propose(ep)
            start = monotonic()
            cp = snapshot(ep, draft)
            cp["context"]["family"] = spec["family"]
            overhead += monotonic()-start
            save(dest.parent / "blind" / f"{cp['step']:03d}.json", cp)
            start = monotonic()
            candidate = choose_card(cp["context"], draft["allowed"], bank)
            selected = False
            if candidate:
                q = protocol["target_hazard"]
                if (spec["id"].startswith("train-") and draft["action"] in candidate["recommended"]):
                    q = protocol.get("training_aligned_hazard", q)
                u = uniform(spec, "checkpoint", cp["step"])
                selected = u < q
                checks.append({"step": cp["step"], "conditional_p": q, "u": u, "selected": selected,
                               "first_hit_p": q*survival, "proposal": draft["action"]})
                survival *= 1-q
            if not selected:
                overhead += monotonic()-start
                cp["execution"] = ep.advance(draft["action"])
                prefix.append(cp)
                continue
            match = assess(cp, candidate)
            row = {"spec": spec, "checkpoint": cp, "match": match, "selection": checks}
            if mode == "paired":
                arms = ["silent", "remind"]
                if uniform(spec, "order", cp["step"]) < .5:
                    arms.reverse()
            else:
                row["assignment"] = select(row, mode, runtime, prefix)
                arms = [row["assignment"]["arm"]]
            overhead += monotonic()-start
            save(dest.with_name("before-execution.json"), {**row, "prefix": prefix, "mode": mode})
            traces, values = {}, {}
            for arm in arms:
                branch = matched(ep, cp, prefix, actor) if mode == "paired" else ep
                try:
                    first = draft if arm == "silent" else actor.propose(branch, match["card"]["reminder"],
                                                                       draft["action"])
                    trace = roll(branch, actor, prefix, first)
                    cost = costs(trace) if arm == "silent" else costs(trace, len(prefix), draft)
                    account(cost, overhead)
                    traces[arm], values[arm] = trace, cost
                finally:
                    if branch is not ep:
                        branch.env.close()
            row.update(arms=values, traces=traces)
            if mode == "paired":
                row["delta"] = {k: values["remind"][k]-values["silent"][k] for k in values["silent"]}
                row["label"] = "remind" if row["delta"]["utility"] > 0 else "silent"
            save(dest, row)
            return row
        trace = {"success": int(ep.won), "steps": prefix}
        cost = costs(trace)
        account(cost, overhead)
        row = {"spec": spec, "match": None, "selection": checks,
               "arms": {"silent": cost}, "traces": {"silent": trace}}
        save(dest, row)
        return row
    finally:
        ep.env.close()


def account(cost, seconds):
    cost["controller_seconds"] = seconds
    cost["seconds"] += seconds
    cost["utility"] -= seconds*PRICES["per_minute"]/60


def probe_fit(rows, budget, token_budget):
    uncertain = [r for r in rows if r["match"] and r["match"]["e"] is None]
    logs = []
    for r in uncertain:
        a = r["assignment"]
        logs.append({**a, "z": int(a["arm"] == "remind"), "fold": intervention_policy.fold_id(r)%4,
                     "r": r["arms"][a["arm"]]["utility"]})
    # Fit using independent development games; keep the last third for feasibility.
    fit_ids = [i for i,r in enumerate(uncertain) if intervention_policy.fold_id(r)%3 != 2]
    train = [logs[i] for i in fit_ids]
    report = probe_value.estimate(train)
    head = probe_value.train_continuation(train)
    report.update(continuation=head, validated=False)
    if head is None or any(min(cell) == 0 for cell in head["observed_cells"]):
        return report
    # Upper-bound harm for a frozen inspect policy with missing counterfactuals.
    held = [(r,l) for i,(r,l) in enumerate(zip(uncertain, logs)) if i not in fit_ids]
    harms, extra = 0, []
    for r,l in held:
        obs = r["arms"][l["arm"]]
        if l["first"] == "inspect":
            q = head["post"][int(l["flag"])]
            wanted = int(q[1] > q[0])
            weight = int(wanted == l["z"])/(l["b1"]*l["b2"])
            harms += int(wanted == 1 and (l["z"] == 0 or not obs["success"]))
        else:
            weight = -int(l["first"] == "silent")/l["b1"]
            harms += int(obs["success"] if l["first"] == "silent" else not obs["success"])
        extra.append(weight*obs["tokens"])
    bound = risk_upper(harms, len(held))
    cost = float(np.mean(extra)+1.96*np.std(extra,ddof=1)/np.sqrt(len(extra))) if len(extra)>1 else None
    report.update(risk_upper=bound, approximate_extra_token_upper=cost,
                  validated=bool(bound <= budget and cost is not None and cost <= token_budget))
    return report
