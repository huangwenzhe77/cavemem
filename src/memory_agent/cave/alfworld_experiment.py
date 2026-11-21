"""Matched ALFWorld interventions, gain-head training and held-out evaluation."""

import random
from collections import Counter
from statistics import mean

import numpy as np

from . import gate_mlp
from .alfworld import MAX_STEPS, PRICES, roll, snapshot
from .alfworld_env import Episode
from .alfworld_memory import applicable, build_bank, matched
from .records import digest, read, save


THRESHOLDS = (0.002, 0.02, 0.10, None)
PHASES = ("holding_target", "holding_other", "target_visible", "search")


def costs(trace, revision_at=None, initial=None):
    drafts = [r["draft"] for r in trace["steps"]]
    if revision_at is not None:
        drafts = [*drafts, initial]
    actor_seconds = sum(d["seconds"] for d in drafts)
    env_seconds = sum(r["execution"]["seconds"] for r in trace["steps"])
    tokens = sum(d["input_tokens"] + d["output_tokens"] for d in drafts)
    steps = len(trace["steps"])
    utility = (trace["success"] - PRICES["per_1000_tokens"] * tokens / 1000
               - PRICES["per_step"] * steps - PRICES["per_minute"] * (actor_seconds + env_seconds) / 60)
    return {"success": trace["success"], "tokens": tokens, "steps": steps,
            "seconds": actor_seconds + env_seconds, "actor_seconds": actor_seconds,
            "env_seconds": env_seconds, "actor_calls": len(drafts), "utility": utility}


def choose_card(context, allowed, bank):
    ordered = sorted(bank, key=lambda card: (("same_target", "same_family", "any").index(
        card["scope"]), card["id"]))
    for card in ordered:
        recommended = applicable(context, allowed, card)
        if recommended:
            return {"card": card, "recommended": recommended}
    return None


def target_game(args, spec, actor, bank, path):
    ep = Episode(args.data, args.config, spec)
    try:
        prefix = []
        while not ep.terminal:
            checkpoint = snapshot(ep, actor.propose(ep))
            context = checkpoint["context"]
            context["family"] = spec["family"]
            draft = checkpoint["draft"]
            match = choose_card(context, draft["allowed"], bank)
            if match is None:
                checkpoint["execution"] = ep.advance(draft["action"])
                prefix.append(checkpoint)
                continue
            match["v"] = match["rho"] = 1
            match["e"] = int(draft["action"] not in match["recommended"])
            save(path.with_name("checkpoint.json"), {"checkpoint": checkpoint, "match": match,
                                                      "prefix": prefix})
            order = ["silent", "remind"]
            random.Random(spec["seed"]).shuffle(order)
            data, traces = {}, {}
            for arm in order:
                other = matched(ep, checkpoint, prefix, actor)
                try:
                    first = draft if arm == "silent" else actor.propose(
                        other, match["card"]["reminder"], draft["action"])
                    trace = roll(other, actor, prefix, first)
                    data[arm] = costs(trace) if arm == "silent" else costs(trace, len(prefix), draft)
                    traces[arm] = trace
                finally:
                    other.env.close()
            delta = {k: data["remind"][k] - data["silent"][k] for k in data["silent"]}
            save(path, {"spec": spec, "checkpoint": checkpoint, "match": match, "order": order,
                        "arms": data, "delta": delta, "traces": traces,
                        "label": "remind" if delta["utility"] > 0 else "silent"})
            print("TARGET_PAIR", spec["id"], checkpoint["step"], match["e"],
                  data["silent"]["success"], data["remind"]["success"], delta["tokens"], flush=True)
            return
        trace = {"success": int(ep.won), "steps": prefix}
        save(path, {"spec": spec, "match": None, "arms": {"silent": costs(trace)},
                    "traces": {"silent": trace}})
        print("TARGET_NO_MATCH", spec["id"], trace["success"], flush=True)
    finally:
        ep.env.close()


def collect(args, manifest, actor, bank, split):
    rows = []
    for spec in manifest["splits"][split]:
        path = args.out / split / spec["id"] / "result.json"
        if not path.exists():
            target_game(args, spec, actor, bank, path)
        rows.append(read(path))
    return rows


def features(row, tokenizer, table):
    cp, match = row["checkpoint"], row["match"]
    ctx, card = cp["context"], match["card"]
    text = "\n".join((ctx["goal"], ctx["observation"], card["reminder"]))
    z = gate_mlp.pool(text, gate_mlp.CONFIG["context_tokens"], tokenizer, table)
    a = gate_mlp.pool(cp["draft"]["action"], gate_mlp.CONFIG["action_tokens"], tokenizer, table)
    return np.concatenate((z, a, [float(ctx["phase"] == p) for p in PHASES],
                           [match["e"], cp["step"] / MAX_STEPS, len(ctx["checked"]) / MAX_STEPS]))


def mean_utility(rows, predictions, threshold):
    if threshold is None:
        return mean(r["arms"]["silent"]["utility"] for r in rows)
    return mean(r["arms"]["remind" if r["match"] and predictions[r["spec"]["id"]] >= threshold
                         else "silent"]["utility"] for r in rows)


def fit(args, actor, splits):
    train = [r for r in splits["train"] if r["match"]]
    val = splits["validation"]
    if len(train) < 4 or not any(r["match"] for r in val):
        return {"status": "insufficient_pairs", "training_pairs": len(train), "threshold": None}
    folder = args.out / "encoder"
    folder.mkdir(exist_ok=True)
    source = gate_mlp.export_encoder(args.weights, folder, actor.info)
    tokenizer, table = gate_mlp.load_encoder(folder, source)
    X = np.stack([features(r, tokenizer, table) for r in train])
    y = np.asarray([r["delta"]["utility"] for r in train])
    model = gate_mlp.train_head(X, y, np.ones(len(y)))
    model.update(status="fit", training_pairs=len(train), encoder=source, hyperparameters=gate_mlp.CONFIG)
    predictions = predict(args, model, val)
    scores = [mean_utility(val, predictions, threshold) for threshold in THRESHOLDS]
    idx = max(range(len(scores)), key=lambda i: (scores[i], i))
    model.update(threshold=THRESHOLDS[idx], validation_grid=list(zip(THRESHOLDS, scores)))
    return model


def predict(args, model, rows):
    if model.get("parameters") is None:
        return {}
    tokenizer, table = gate_mlp.load_encoder(args.out / "encoder", model["encoder"])
    params = [np.asarray(p) for p in model["parameters"]]
    return {r["spec"]["id"]: float(gate_mlp.forward(features(r, tokenizer, table), params)[1])
            for r in rows if r["match"]}


def summary(rows, model=None, predictions=None):
    pairs = [r for r in rows if r["match"]]
    outcomes = Counter()
    for r in pairs:
        a, b = (r["arms"][arm]["success"] for arm in ("silent", "remind"))
        key = {(0, 0): "both_fail", (0, 1): "rescued", (1, 0): "harmed", (1, 1): "both_success"}[a, b]
        outcomes[key] += 1
        if key == "both_success":
            for cost in ("tokens", "seconds", "steps"):
                outcomes[f"both_success_more_{cost}"] += r["delta"][cost] > 0
            outcomes["both_success_more_tokens_and_seconds"] += (
                r["delta"]["tokens"] > 0 and r["delta"]["seconds"] > 0)
    policies = {}
    for mode in ("silent", "always_when_applicable", "gain_gate"):
        if mode == "gain_gate" and model is None:
            continue
        threshold = model["threshold"] if model else None
        intervention_ids, selected = [], []
        for r in rows:
            remind = bool(r["match"]) and (
                mode == "always_when_applicable" or
                (mode == "gain_gate" and threshold is not None
                 and predictions[r["spec"]["id"]] >= threshold))
            if remind:
                intervention_ids.append(r["spec"]["id"])
            selected.append(r["arms"]["remind" if remind else "silent"])
        policies[mode] = {"reminders": len(intervention_ids), "reminder_ids": intervention_ids,
                          "wins": sum(x["success"] for x in selected),
                          "means": {k: mean(x[k] for x in selected) for k in selected[0]}}
    return {"games": len(rows), "pairs": len(pairs), "outcomes": dict(outcomes),
            "original_aligned": sum(r["match"]["e"] == 0 for r in pairs),
            "labels": dict(Counter(r["label"] for r in pairs)), "policies": policies,
            "mean_pair_delta": {k: mean(r["delta"][k] for r in pairs) for k in pairs[0]["delta"]}
            if pairs else {}}


def run(args, manifest, actor):
    if manifest["policy"].get("controller_version") == 2:
        return run_controller(args, manifest, actor)
    bank = build_bank(args, manifest, actor)
    splits = {split: collect(args, manifest, actor, bank, split)
              for split in ("train", "validation")}
    dest = args.out / "gain-gate.json"
    provenance = {"bank": digest(bank), "manifest": digest(manifest),
                  "development": digest({s: [r["arms"] for r in rows]
                                         for s, rows in splits.items()})}
    if dest.exists():
        model = read(dest)
        if model["provenance"] != provenance:
            raise ValueError("gate inputs changed after freezing")
    else:
        model = fit(args, actor, splits)
        save(dest, {**model, "provenance": provenance})
    train_labels = [{"id": r["spec"]["id"], "match": r["match"], "checkpoint": r["checkpoint"],
                     "label": r["label"], "delta": r["delta"]} for r in splits["train"] if r["match"]]
    save(args.out / "train-labels.json", train_labels)
    print("FROZEN_GATE", model["status"], model["threshold"], flush=True)
    splits["test"] = collect(args, manifest, actor, bank, "test")
    report = {s: summary(rows, model, predict(args, model, rows)) for s, rows in splits.items()}
    source_results = [read(p) for p in sorted((args.out / "source").glob("*/replay-*.json"))]
    source_natural = [read(p) for p in sorted((args.out / "source").glob("*/natural.json"))]
    report["source"] = {"games": len(source_natural), "wins": sum(r["success"] for r in source_natural),
                        "replayed_candidates": len(source_results), "admitted": len(bank),
                        "direct_wins": sum(r["branches"]["direct"]["success"] for r in source_results),
                        "remind_wins": sum(r["branches"]["remind"]["success"] for r in source_results)}
    save(args.out / "summary.json", report)
    print("COMPLETE", {s: (v["games"], v.get("pairs")) for s, v in report.items()}, flush=True)


def controller_cohort(args, manifest, actor, bank, split, mode, runtime=None):
    from .alfworld_controller import collect_one

    if manifest["policy"].get("holdout_mechanism") and bank:
        from .alfworld_controller import pattern

        def mechanism(card):
            return (card["phase"], pattern(card["source_original"]), card["pattern"])
        keys = sorted({mechanism(c) for c in bank}, key=digest)
        reserved = keys[0]
        if len(keys) < 2:
            raise ValueError("mechanism holdout needs at least two verified failure signatures")
        bank = [c for c in bank if (mechanism(c) == reserved) == (split == "test")]
        save(args.out / "controller" / "mechanism-split.json",
             {"test": reserved, "development": keys[1:], "definition": "phase, source action, correction pattern"})
    rows = []
    for spec in manifest["splits"][split]:
        dest = args.out / "controller" / split / mode / spec["id"] / "result.json"
        binding = {"spec": digest(spec), "actor": digest(actor.info), "manifest": digest(manifest),
                   "bank": digest(bank), "mode": mode, "model": digest(runtime[0]) if runtime else None}
        receipt = dest.with_name("inputs.json")
        if receipt.exists():
            if read(receipt) != binding:
                raise ValueError("frozen rollout inputs changed")
        else:
            if dest.exists():
                raise ValueError("existing rollout missing input provenance")
            save(receipt, binding)
        row = read(dest) if dest.exists() else collect_one(args, spec, actor, bank, manifest["policy"],
                                                          mode, dest, runtime)
        rows.append(row)
    return rows


def train_controller(args, manifest, actor, bank, train_rows, val_rows):
    from . import alfworld_controller as controller, intervention_policy

    config = manifest["policy"]
    provenance = {"manifest": digest(manifest), "bank": digest(bank), "actor": digest(actor.info),
                  "train": digest(train_rows), "validation": digest(val_rows)}
    dest = args.out / "controller" / "frozen.json"
    if dest.exists():
        model = read(dest)
        if (model["provenance"] != provenance or
                model.get("controller_id") != digest({k:v for k,v in model.items() if k != "controller_id"})):
            raise ValueError("controller training inputs changed")
        return model
    encoder_dir = args.out / "controller" / "encoder"
    source, encoder = None, None
    if sum(bool(r["match"]) for r in train_rows) >= 4:
        encoder_dir.mkdir(parents=True, exist_ok=True)
        source = gate_mlp.export_encoder(args.weights, encoder_dir, actor.info)
        encoder = gate_mlp.load_encoder(encoder_dir, source)
        def vectors(row):
            return controller.vector(row, *encoder)
        model = intervention_policy.train(train_rows, vectors, config)
        model["calibration"] = intervention_policy.calibrate(model, val_rows, vectors)
    else:
        model = {"status": "insufficient_training", "calibration": {
                 "enabled": False, "reason": "insufficient_training"}}
    train_contexts = {controller.decision_context(r) for r in train_rows if r["match"]}
    val_contexts = {controller.decision_context(r) for r in val_rows if r["match"]}
    model["validated_contexts"] = sorted(train_contexts & val_contexts)
    if "diagnostic" in manifest["splits"]:
        diagnostic = controller_cohort(args, manifest, actor, bank, "diagnostic", "diagnostic", (model, encoder))
        model["probe"] = controller.probe_fit(diagnostic, config["risk_budget"], config["extra_token_budget"])
        model["diagnostic_digest"] = digest(diagnostic)
    model.update(provenance=provenance, encoder=source)
    model["controller_id"] = digest(model)
    save(dest, model)
    return model


def deploy(args, manifest, actor):
    from .alfworld_controller import load_bank

    if manifest["policy"].get("controller_version") != 2:
        raise ValueError("deployment requires a cohort registered with the learned controller")
    model = read(args.out / "controller" / "frozen.json")
    if digest({k:v for k,v in model.items() if k != "controller_id"}) != model.get("controller_id"):
        raise ValueError("frozen controller digest mismatch")
    bank = load_bank(args.out, read(args.out / "bank.json"))
    expected = model["provenance"]
    if (expected["manifest"], expected["actor"], expected["bank"]) != (
            digest(manifest), digest(actor.info), digest(bank)):
        raise ValueError("deployment artifacts differ from training")
    encoder = gate_mlp.load_encoder(args.out / "controller" / "encoder", model["encoder"]) if model["encoder"] else None
    rows = controller_cohort(args, manifest, actor, bank, "test", "live", (model, encoder))
    costs = [next(iter(r["arms"].values())) for r in rows]
    stats = {"games": len(rows), "wins": sum(c["success"] for c in costs),
             "reminders": sum("remind" in r["arms"] for r in rows),
             "inspections": sum(bool(r.get("assignment", {}).get("inspected")) for r in rows),
             "eligible": sum(bool(r["match"]) for r in rows),
             "means": {k: mean(c[k] for c in costs) for k in costs[0]} if rows else {}}
    save(args.out / "controller" / "live-summary.json", stats)
    return stats


def run_controller(args, manifest, actor):
    from .alfworld_controller import load_bank

    bank = load_bank(args.out, build_bank(args, manifest, actor))
    mode = manifest["policy"]["train_collection"]
    if mode not in ("paired", "single_arm"):
        raise ValueError("invalid registered collection mode")
    train_rows = controller_cohort(args, manifest, actor, bank, "train", mode)
    val_rows = controller_cohort(args, manifest, actor, bank, "validation", mode)
    model = train_controller(args, manifest, actor, bank, train_rows, val_rows)
    labels = [{"spec": r["spec"]["id"], "checkpoint": r["checkpoint"], "match": r["match"],
               "label": r["label"], "delta": r["delta"]} for r in train_rows if "label" in r]
    save(args.out / "train-labels.json", labels)
    report = {"live": deploy(args, manifest, actor), "model_status": model["status"],
              "source_cards": len(bank), "collection": mode,
              "train_eligible": sum(bool(r["match"]) for r in train_rows),
              "validation_eligible": sum(bool(r["match"]) for r in val_rows)}
    if mode == "paired":
        audit_rows = controller_cohort(args, manifest, actor, bank, "test", "paired")
        report["paired"] = summary(audit_rows)
        report["paired_comparisons"] = compare_policies(args, model, audit_rows)
    else:
        report["logged_validation"] = {"assigned": sum("assignment" in r for r in val_rows),
                                       "games": len(val_rows), "calibration": model["calibration"]}
    evidence = [read(p) for p in sorted((args.out / "source").glob("*/replay-*.json"))]
    replicates = [rep for item in evidence for rep in item["replicates"]]
    report["source"] = {"candidates": len(evidence), "paired_replicates": len(replicates),
                        "admitted": len(bank), "successes_per_arm": {
                            a: sum(r["metrics"][a]["success"] for r in replicates)
                            for a in ("silent", "direct", "remind")}}
    save(args.out / "summary.json", report)
    return report


def compare_policies(args, model, rows):
    """Task-level estimates on identical held-out forks, with all unmatched games retained."""
    from . import intervention_policy, constrained_gate
    from .alfworld_controller import in_domain, vector

    names = ("silent", "applicable", "exposure_rule", "supervised", "constrained")
    estimates = {k: [] for k in names}
    encoder = (gate_mlp.load_encoder(args.out / "controller" / "encoder", model["encoder"])
               if model.get("encoder") else None)
    for r in rows:
        probs = dict.fromkeys(names, 0.)
        silent = r["arms"]["silent"]
        remind = r["arms"].get("remind", silent)
        if r["match"]:
            probs["applicable"] = 1.
            probs["exposure_rule"] = float(r["match"]["e"] == 1)
            if model["status"] == "fit":
                x = vector(r, *encoder)
                sft = [np.asarray(p) for p in model["warm_start"]["parameters"]]
                probs["supervised"] = float(constrained_gate.sigmoid(gate_mlp.forward(x, sft)[1]))
                if in_domain(model, r):
                    probs["constrained"] = intervention_policy.decide(model, x, 0.)["probability"]
        for name, p in probs.items():
            metrics = {k: silent[k] + p*(remind[k]-silent[k]) for k in ("utility", "success", "tokens", "seconds")}
            estimates[name].append({**metrics, "p": p, "delta_utility": p*(remind["utility"]-silent["utility"]),
                                    "harm": p*int(silent["success"] > remind["success"]),
                                    "rescue": p*int(silent["success"] < remind["success"])})
    out = {}
    for name, metrics in estimates.items():
        if not metrics:
            continue
        # Same draws across policies. One registered task contributes one row.
        gain = np.array([x["delta_utility"] for x in metrics])
        rng = np.random.default_rng(4261)
        draws = np.array([rng.choice(gain, size=len(gain)).mean() for _ in range(1000)])
        out[name] = {"n": len(metrics), "means": {k: mean(x[k] for x in metrics) for k in metrics[0]},
                     "task_bootstrap_utility_delta_ci95": np.quantile(draws, [.025,.975]).tolist()}
    return {"scope": "matched branches and unmatched natural tasks, binary decisions; "
                     "policy-specific online overhead is measured by the separate live rollout",
            "policies": out, "gated_probe_included": False}
