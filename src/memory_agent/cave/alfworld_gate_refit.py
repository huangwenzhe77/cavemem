"""Train ALFWorld reminder heads and a constrained gate from archived paired rollouts."""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from memory_agent.cave import constrained_gate
from memory_agent.cave.alfworld_env import sha_file
from memory_agent.cave.alfworld_full_study import available
from memory_agent.cave.alfworld_head_pilot import FEATURES, decision_features, evaluate, fit
from memory_agent.cave.alfworld_study_rollout import receipt
from memory_agent.cave.alfworld_transfer_study import audit_pair, score_tokens
from memory_agent.cave.records import read, save


SEED = 262734
LIMITS = {"policy_risk_delta": 0.05, "policy_mean_extra_tokens_budget": 256}


def register(args, frozen, manifest):
    if args.out.exists():
        raise FileExistsError(args.out)
    eval_specs = [s for group in ("valid_seen", "valid_unseen") for s in manifest["splits"][group]]
    upstream = [s for group in ("historical", "development", "demonstration") for s in manifest["splits"][group]]
    historical_templates = {spec["template"] for spec in upstream}
    grouped = defaultdict(list)
    for spec in eval_specs:
        grouped[spec["template"]].append(spec["id"])
    rng = random.Random(SEED)
    eligible = sorted(set(grouped) - historical_templates)
    rng.shuffle(eligible)
    validation = []
    for template in eligible:
        validation.extend(grouped[template])
        if len(validation) >= 54:
            break
    if len(validation) < 54:
        raise ValueError("insufficient validation games outside the upstream templates")
    training = sorted({s["id"] for s in eval_specs} - set(validation))
    registrations = sorted([*args.study.parent.glob("cave-alfworld*/manifest.json"),
                            *args.study.parent.glob("cave-alfworld*/sealed-test.json")])
    seen = [s for path in registrations
            for records in read(path).get("splits", {}).values() for s in records]
    seen += eval_specs + upstream
    already = {key: {s.get(key) for s in seen} for key in ("template", "game", "sha256")}
    fresh = [
        row for row in available(args.data, "train")
        if row["template"] not in already["template"] and row["file"] not in already["game"]
        and row["file_sha256"] not in already["sha256"]
    ]
    rng.shuffle(fresh)
    by_template = {}
    for row in fresh:
        by_template.setdefault(row["template"], row)
    per_family = defaultdict(list)
    for row in by_template.values():
        per_family[row["family"]].append(row)
    candidates = [rows[i] for i in range(max(map(len, per_family.values())))
                  for _, rows in sorted(per_family.items()) if i < len(rows)]
    if len(candidates) < 150:
        raise ValueError("insufficient never-registered test templates")
    test_specs = [
        {"id": f"gate-test-{i:03d}", "game": row["file"], "sha256": row["file_sha256"],
         "template": row["template"], "family": row["family"], "seed": rng.randrange(2**32)}
        for i, row in enumerate(candidates[:150], 1)
    ]
    save(args.out / "sealed-test.json", {"splits": {"test": test_specs}, "seed": SEED,
                                         "config_sha256": manifest["config_sha256"],
                                         "outcomes_opened": False,
                                         "source": "official train directory; never-registered game, hash and template"})
    save(args.out / "cohort.json", {"seed": SEED, "train": training, "validation": sorted(validation),
                                     "groups": {s["id"]: s["template"] for s in eval_specs},
                                     "transfer_sha256": sha_file(args.study / "transfer-freeze.json"),
                                     "sealed_test_sha256": sha_file(args.out / "sealed-test.json"),
                                     "budgets": frozen["token_budgets"], "limits": LIMITS,
                                     "compute_scope": "additional actor tokens per game",
                                     "policy_bootstrap_unit": "template"})
    print("REGISTERED", len(training), len(validation), len(test_specs),
          dict(Counter(s["family"] for s in test_specs)), flush=True)


def paired_rows(study, frozen, manifest, registry, feature_builder=decision_features):
    bank_path = study / "transfer-bank.json"
    if sha_file(bank_path) != frozen["bank_sha256"]:
        raise ValueError("changed transfer bank")
    cards = {c["id"]: c for c in read(bank_path)["cards"]}
    base = Path(frozen["baseline"])
    vocabulary = set(read(base / "ranker.json")["vocabulary"])
    specs = {s["id"]: s for key in ("valid_seen", "valid_unseen") for s in manifest["splits"][key]}
    train, valid = map(set, (registry["train"], registry["validation"]))
    if train & valid or train | valid != set(specs):
        raise ValueError("overlapping or incomplete old-cohort split")
    if {specs[k]["template"] for k in train} & {specs[k]["template"] for k in valid}:
        raise ValueError("template overlap between training and validation")
    roots = SimpleNamespace(out=study)
    dataset = {}
    for split in ("train", "validation"):
        records = []
        for game in registry[split]:
            spec = specs[game]
            natural = read(receipt(base, spec))
            pair = read(receipt(study, spec, "paired"))
            branches = audit_pair(roots, spec, natural, pair, set(cards))
            for setting, budget in [("calls", None), *registry["budgets"].items()]:
                outcome = pair if budget is None else score_tokens(natural, pair["match"], branches, budget)
                match = outcome["match"]
                silent, remind = outcome["silent"], outcome["episodic"]
                features = feature_builder(natural, match, cards[match["card"]], vocabulary, budget) if match else None
                records.append({"game": game, "template": spec["template"], "setting": setting,
                                "strict": bool(match and match["strict_contract"]), "features": features,
                                "silent": silent, "remind": remind, "D": remind["utility"] - silent["utility"],
                                "H": int(silent["success"] and not remind["success"]),
                                "extra": remind["tokens"] - silent["tokens"]})
        dataset[split] = records
    return dataset


def arrays(rows, center, scale):
    return ((torch.tensor([r["features"] for r in rows], dtype=torch.float32) - center) / scale)


def fit_policy(rows, heads, center, scale, limits):
    matched = [r for r in rows if r["features"] is not None]
    x = arrays(matched, center, scale)
    gain, harm, extra = (np.array([r[key] for r in matched]) for key in ("D", "H", "extra"))
    templates = [r["template"] for r in matched]
    counts = Counter(templates)
    weight = torch.tensor([1 / counts[key] for key in templates], dtype=x.dtype)
    weight /= weight.sum()
    warm = nn.Sequential(nn.Linear(len(FEATURES), 8), nn.Tanh(), nn.Linear(8, 1))
    temp = constrained_gate.CONFIG["temperature"]
    with torch.no_grad():
        warm[0].load_state_dict(heads[0].state_dict())
        warm[2].weight.copy_(heads[2].weight[:1] / temp)
        warm[2].bias.copy_(heads[2].bias[:1] / temp)
    labels = torch.tensor((gain > 0) & (harm == 0), dtype=x.dtype)
    optimizer = torch.optim.Adam(warm.parameters(), lr=0.01, weight_decay=0.005)
    for _ in range(200):
        optimizer.zero_grad()
        loss = (weight * F.binary_cross_entropy_with_logits(warm(x)[:, 0], labels, reduction="none")).sum()
        loss.backward()
        optimizer.step()
    p = [warm[0].weight.detach().numpy().T.copy(), warm[0].bias.detach().numpy().copy(),
         warm[2].weight.detach().numpy()[0].copy() * temp, warm[2].bias.detach().numpy().copy() * temp]
    return constrained_gate.optimize(x.numpy(), gain, harm, extra, p, 0, limits, constrained_gate.CONFIG["kl"],
                                     clusters=templates)


def policy_report(rows, decisions):
    totals = defaultdict(Counter)
    for row, remind in zip(rows, decisions, strict=True):
        outcome = row["remind"] if remind else row["silent"]
        counts = {"games": 1, "reminders": int(remind), "wins": outcome["success"],
                  "harms": int(remind and row["H"]),
                  "rescues": int(remind and outcome["success"] and not row["silent"]["success"]),
                  "extra_tokens": int(row["extra"]) if remind else 0, "tokens": outcome["tokens"],
                  "utility": outcome["utility"]}
        totals[row["setting"]].update(counts)
        totals["overall"].update(counts)
    return dict(totals)


def validate(rows, heads, policy, center, scale, limits):
    matched = np.array([r["features"] is not None for r in rows])
    x = arrays([r for r in rows if r["features"] is not None], center, scale)
    with torch.no_grad():
        values = heads(x)
        gain = np.full(len(rows), -np.inf)
        risk = np.full(len(rows), 1.0)
        prob = np.zeros(len(rows))
        gain[matched] = values[:, 0].numpy()
        risk[matched] = values[:, 1].sigmoid().numpy()
        _, logits = constrained_gate.gate_mlp.forward(x.numpy(), [np.asarray(p) for p in policy["parameters"]])
        prob[matched] = constrained_gate.sigmoid(logits)
    reports = {
        "silent": policy_report(rows, np.zeros(len(rows), dtype=bool)), "always_remind": policy_report(rows, matched),
        "predictors_fixed": policy_report(rows, matched & (gain > 0) & (risk <= .20)),
        "trained_policy_fixed": policy_report(rows, matched & (prob >= .5)),
    }
    templates = [r["template"] for r in rows]
    _, assignment, count = np.unique(templates, return_inverse=True, return_counts=True)
    n = len(count)
    draws = np.random.default_rng(SEED).multinomial(n, np.full(n, 1/n), size=1024) / n
    widths = draws[:, assignment] / count[assignment]
    harm, cost, value = (np.array([r[key] for r in rows]) for key in ("H", "extra", "D"))
    candidates = (
        ("predictors", {"gain_cutoff": cutoff, "risk_limit": ceiling}, matched & (gain > cutoff) & (risk <= ceiling))
        for cutoff in (0., .1, .25) for ceiling in (.1, .2, .35)
    )
    selected = {}
    for kind, config, decision in [*candidates, *(
        ("trained_policy", {"probability_cutoff": cutoff}, matched & (prob >= cutoff))
        for cutoff in (.25, .5, .75, .9, .95, .99)
    )]:
        risk_upper = float(np.quantile(widths @ (harm * decision), .95))
        cost_upper = float(np.quantile(widths @ (cost * decision), .95))
        utility = float(np.sum(value * decision / count[assignment]) / n)
        if (risk_upper <= limits["policy_risk_delta"] and cost_upper <= limits["policy_mean_extra_tokens_budget"]
                and utility > selected.get(kind, {}).get("utility_improvement", 0)):
            selected[kind] = {"config": config, "risk_upper": risk_upper, "extra_tokens_upper": cost_upper,
                              "utility_improvement": utility, "comparison": policy_report(rows, decision)}
    for kind in ("predictors", "trained_policy"):
        selected.setdefault(kind, {"config": {"always_silent": True}, "comparison": reports["silent"]})
    return {"fixed_policy_comparison": reports, "validation_selected": selected,
            "validation_harm_games": len({r["game"] for r in rows if r["H"]}),
            "validation_rescue_games": len({r["game"] for r in rows
                                            if r["remind"]["success"] and not r["silent"]["success"]})}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("register", "fit"))
    for name in ("out", "study", "data"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args()
    torch.set_num_threads(1)
    frozen = read(args.study / "transfer-freeze.json")
    manifest_path = Path(frozen["baseline"]) / "manifest.json"
    if sha_file(manifest_path) != frozen["manifest_sha256"]:
        raise ValueError("changed baseline manifest")
    manifest = read(manifest_path)
    if args.stage == "register":
        return register(args, frozen, manifest)
    cohort = read(args.out / "cohort.json")
    if (sha_file(args.out / "sealed-test.json") != cohort["sealed_test_sha256"]
            or sha_file(args.study / "transfer-freeze.json") != cohort["transfer_sha256"]):
        raise ValueError("changed frozen cohort input")
    source = paired_rows(args.study, frozen, manifest, cohort)
    save(args.out / "paired-labels.json", source)
    heads, center, scale, trained_risk, stats = fit(source["train"])
    if not trained_risk:
        raise ValueError("insufficient train harm cases for a risk head")
    print("HEADS_TRAINED", stats, flush=True)
    policy = fit_policy(source["train"], heads, center, scale, cohort["limits"])
    validation = validate(source["validation"], heads, policy, center, scale, cohort["limits"])
    save(args.out / "gate-model.json", {"features": FEATURES, "center": center.tolist(), "scale": scale.tolist(),
                                        "heads": {k: v.tolist() for k, v in heads.state_dict().items()},
                                        "constrained": policy, "limits": cohort["limits"],
                                        "decision": {k: v["config"] for k, v in validation["validation_selected"].items()},
                                        "cohort_sha256": sha_file(args.out / "cohort.json"),
                                        "labels_sha256": sha_file(args.out / "paired-labels.json")})
    prediction_error = evaluate(source["validation"], heads, center, scale, trained_risk, stats)["prediction_error"]
    save(args.out / "validation-report.json", {"training": stats, "validation": validation,
                                                "prediction_error": prediction_error})
    print("POLICY_TRAINED", policy["train_constraints_met"], policy["trace"][-1], flush=True)
    print("VALIDATION_SELECTED", {k: v["config"] for k, v in validation["validation_selected"].items()}, flush=True)


if __name__ == "__main__":
    main()
