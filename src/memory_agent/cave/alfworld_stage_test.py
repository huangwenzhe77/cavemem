"""Seal an independent cohort, then test one frozen stage policy by paired contrast."""

import argparse
import os
import random
import sys
from pathlib import Path

from memory_agent.cave.alfworld_env import sha_file
from memory_agent.cave.alfworld_full_study import available
from memory_agent.cave.alfworld_gate_refit import policy_report
from memory_agent.cave.alfworld_gate_test import interval
from memory_agent.cave.alfworld_memory_effects import attach
from memory_agent.cave.alfworld_stage_pairs import collect_one, configuration
from memory_agent.cave.alfworld_stage_policy import decide, predict
from memory_agent.cave.alfworld_study_pairs import cached, context_applies
from memory_agent.cave.alfworld_study_rollout import receipt, rollout
from memory_agent.cave.alfworld_transfer_study import (
    NEUTRAL, audit_pair, choose_transfer, continuing_advice, episodic_note, score_tokens,
)
from memory_agent.cave.records import read, save


SEED = 262738
PRIMARY = "p50"


def register(args):
    path = args.out / "sealed-test.json"
    if path.exists():
        raise FileExistsError(path)
    roots = sorted(args.study.parent.glob("cave-alfworld*"))
    transfer = read(args.study / "transfer-freeze.json")
    base_manifest = read(Path(transfer["baseline"]) / "manifest.json")
    # Exclude every pre-existing game. Do not use the previous holdout outcomes in training.
    upstream, previous_tests = [], []
    for root in roots:
        for name in ("manifest.json", "sealed-test.json"):
            file = root / name
            if file.exists():
                into = previous_tests if name == "sealed-test.json" else upstream
                into.extend(s for rows in read(file).get("splits", {}).values() for s in rows)
    prior = upstream + previous_tests
    seen_paths = {s.get("game") for s in prior}
    seen_hashes = {s.get("sha256") for s in prior}
    upstream_groups = {s.get("template") for s in upstream}
    previous_groups = {s.get("template") for s in previous_tests}
    pool = [item for item in available(args.data, "train")
            if item["file"] not in seen_paths and item["file_sha256"] not in seen_hashes
            and item["template"] not in upstream_groups]
    rng = random.Random(SEED)
    rng.shuffle(pool)
    groups = {}
    for s in pool:
        groups.setdefault(s["template"], s)
    if len(groups) < 150:
        raise ValueError("not enough holdout templates disjoint from training and validation")
    test = [{"id": f"stage-test-{i:03d}", "template": s["template"], "family": s["family"],
             "game": s["file"], "sha256": s["file_sha256"], "seed": rng.randrange(2**32),
             "entirely_new_template": s["template"] not in previous_groups}
            for i, s in enumerate(groups.values(), 1)]
    save(path, {"splits": {"test": test}, "seed": SEED, "primary_budget": PRIMARY, "outcomes_opened": False,
                "config_sha256": base_manifest["config_sha256"],
                "transfer_sha256": sha_file(args.study / "transfer-freeze.json")})
    print("SEALED_STAGE_TEST", len(test), sum(s["entirely_new_template"] for s in test), flush=True)


def summarize(rows, model):
    forecasts = predict(rows, model["model"])
    linear = predict(rows, model["linear"])
    action = {
        "silent": [False] * len(rows), "always_remind": [r["features"] is not None for r in rows],
        "mlp": decide(rows, forecasts, model["rules"]), "linear": decide(rows, linear, model["linear_rules"]),
    }
    counts = {name: policy_report(rows, chosen) for name, chosen in action.items()}
    primary = [i for i, r in enumerate(rows) if r["setting"] == PRIMARY]
    used = [rows[i] for i in primary]
    evidence = {}
    for name in ("silent", "always_remind", "linear"):
        delta = [int(action["mlp"][i]) - int(action[name][i]) for i in primary]
        evidence[name] = {"success_gain": interval(used, [a * (r["R"] - r["H"]) for a, r in zip(delta, used, strict=True)]),
                          "token_saving": interval(used, [-a * r["extra"] for a, r in zip(delta, used, strict=True)])}
    gate = counts["mlp"][PRIMARY]
    point = all(gate["wins"] >= counts[k][PRIMARY]["wins"] and gate["tokens"] <= counts[k][PRIMARY]["tokens"]
                for k in ("silent", "always_remind"))
    stronger = (point and gate["rescues"] > 0 and gate["harms"] < counts["always_remind"][PRIMARY]["harms"]
                and evidence["silent"]["success_gain"]["ci95"][0] > 0
                and all(evidence[k]["token_saving"]["ci95"][0] > 0 for k in ("silent", "always_remind")))
    return {"policies": counts, "paired_template_intervals": evidence, "meets_point_goal": point,
            "demonstrated_goal": stronger}


def legacy_rows(args, spec, actor, bank, budgets, natural, memory_order):
    candidate = choose_transfer(actor, natural["trace"]["steps"], bank)
    baseline = natural["metrics"]
    paired = {"spec": spec["id"], "match": None, "silent": baseline, "episodic": baseline, "neutral": baseline}
    if candidate is not None:
        anchor, card, analogous = candidate
        step = len(anchor["prefix"]) + 1
        shortlist = anchor["draft"]["allowed"]
        match = {"step": step, "card": card["id"], "state": anchor["state"], "original": anchor["draft"]["action"],
                 "analogous": analogous, "strict_contract": context_applies(actor, anchor, analogous, card),
                 "source_action_legal_in_shortlist": card["source_action"] in shortlist, "evidence": {}}
        paired["match"] = match
        notice = episodic_note(card, shortlist, args.representation)
        continuation = continuing_advice(card, step - 1, args)
        path = receipt(args.out, spec, "episodic")
        episode = cached(path, lambda: rollout(args, spec, actor, anchor=anchor, reminder=notice,
                                               continuation_reminder=continuation))
        paired["episodic"] = episode["metrics"]
        match["evidence"]["episodic"] = {"file": path.name, "sha256": sha_file(path)}
        path = receipt(args.out, spec, "neutral-first")
        neutral = cached(path, lambda: rollout(args, spec, actor, anchor=anchor, reminder=NEUTRAL, cap=step + 1))
        paired["neutral"] = neutral["metrics"]
        match["evidence"]["neutral"] = {"file": path.name, "sha256": sha_file(path)}
    arms = audit_pair(args, spec, natural, paired, set(memory_order))
    rows = []
    for setting, budget in [("calls", None), *budgets.items()]:
        outcome = paired if budget is None else score_tokens(natural, paired["match"], arms, budget)
        match = outcome["match"]
        s, r = outcome["silent"], outcome["episodic"]
        features = attach(natural, match, card, actor.ranker.vocabulary, budget, memory_order) if match else None
        rows.append({"game": spec["id"], "template": spec["template"], "family": spec["family"], "setting": setting,
                     "features": features, "D": r["utility"] - s["utility"],
                     "R": int(r["success"] > s["success"]), "H": int(s["success"] > r["success"]),
                     "extra": r["tokens"] - s["tokens"], "silent": s, "remind": r})
    return {"rows": rows}


def evaluate(args):
    test_path = args.out / "sealed-test.json"
    test = read(test_path)
    model_path = args.development / "model.json"
    model = read(model_path)
    source = read(args.development / "collector-freeze.json")
    transfer_path = args.study / "transfer-freeze.json"
    if (test["transfer_sha256"] != sha_file(transfer_path) or test["config_sha256"] != sha_file(args.config)
            or model["collector_sha256"] != sha_file(args.development / "collector-freeze.json")
            or model["dataset_sha256"] != sha_file(args.development / "paired-labels.json")):
        raise ValueError("frozen stage model or test manifest changed")
    if args.natural_from is not None:
        parent = args.natural_from
        if (sha_file(parent / "sealed-test.json") != sha_file(test_path)
                or read(parent / "test-freeze.json")["manifest_sha256"] != sha_file(test_path)
                or not (parent / "test-report.json").is_file()):
            raise ValueError("archived natural arm does not match the sealed test")
    args.selection = source["selection"]
    protocol = {"manifest_sha256": sha_file(test_path), "model_sha256": sha_file(model_path),
                "code_sha256": sha_file(Path(__file__)), "rules": model["rules"], "primary": PRIMARY,
                "natural_from": str(args.natural_from.resolve()) if args.natural_from else None}
    freeze_path = args.out / "test-freeze.json"
    if freeze_path.exists() and read(freeze_path) != protocol:
        raise ValueError("test already began with another model")
    save(freeze_path, protocol)
    transfer, _, _, _, actor, bank = configuration(args)
    args.representation = transfer["representation"]
    args.exposure = transfer["exposure"]
    rows = []
    seen = test["splits"]["test"]
    newly_collected = 0
    for i, spec in enumerate(seen, 1):
        score_path = receipt(args.out, spec, "scored")
        existing = score_path.exists()
        natural = (read(receipt(args.natural_from, spec)) if args.natural_from is not None
                   else cached(receipt(args.out, spec), lambda: rollout(args, spec, actor)))
        if args.selection == "legacy-wide-persistent":
            result = cached(score_path, lambda: legacy_rows(
                args, spec, actor, bank, transfer["token_budgets"], natural, source["memory_order"]
            ))
        else:
            result = cached(score_path,
                            lambda: collect_one(args, spec, actor, bank, transfer["token_budgets"], natural))
        rows.extend({**r, "entirely_new_template": spec["entirely_new_template"]} for r in result["rows"])
        if not existing:
            newly_collected += 1
            save(args.out / "progress.json", {"completed": i, "total": len(seen)})
            print("STAGE_TEST", i, len(seen), spec["id"], flush=True)
        if newly_collected == 72 and i < len(seen):
            # TextWorld maps one temporary planner library per environment reset until process exit.
            print("RESTART_TO_RELEASE_PLANNER_MAPPINGS", i, flush=True)
            os.execv(sys.executable, args.restart_command)
    report = summarize(rows, model)
    entirely_new = [r for r in rows if r["entirely_new_template"]]
    save(args.out / "test-report.json", {"games": len(seen), "primary": PRIMARY, **report,
                                        "entirely_new_templates": summarize(entirely_new, model) if entirely_new else None})
    print("STAGE_TEST_COMPLETE", len(seen), report["demonstrated_goal"], flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("register", "test"))
    for name in ("out", "study", "refit", "data", "config", "weights", "development"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--natural-from", type=Path)
    args = cli.parse_args()
    args.restart_command = [sys.executable, "-u", "-m", __spec__.name, *sys.argv[1:]]
    (register if args.stage == "register" else evaluate)(args)


if __name__ == "__main__":
    main()
