"""Repeat the locked short-reminder gate on fresh instances with a live token cap."""

import argparse
import os
import sys
from pathlib import Path

import torch

from memory_agent.cave.alfworld_env import sha_file
from memory_agent.cave.alfworld_placement_test import BudgetActor, SETTING, precommit, run, summarize
from memory_agent.cave.alfworld_stage_pairs import choose_stage, configuration, stage_features
from memory_agent.cave.alfworld_stage_policy import fit, predict, select
from memory_agent.cave.alfworld_study_pairs import anchor_at, cached, rendered
from memory_agent.cave.alfworld_study_rollout import receipt
from memory_agent.cave.alfworld_transfer_study import score_tokens, token_prefix
from memory_agent.cave.records import read, save


ACTION_FIELDS = {0, 2, 3, 4, 9}


def mask(rows):
    return [{**r, "features": [v if i not in ACTION_FIELDS else 0 for i, v in enumerate(r["features"])]}
            if r["features"] is not None else r for r in rows]


def lock(args):
    seal = args.out / "sealed-test.json"
    model_path = args.development / "model.json"
    source_data = args.development / "paired-labels.json"
    original = read(model_path)
    if original["dataset_sha256"] != sha_file(source_data):
        raise ValueError("changed short-reminder training data")
    previous_path = args.previous / "test-freeze.json"
    if read(previous_path)["model_sha256"] != sha_file(model_path):
        raise ValueError("extension uses a different model from the first evaluation")
    protocol = {"seal": sha_file(seal), "model": sha_file(model_path), "previous": sha_file(previous_path),
                "code": {p: sha_file(Path(__file__).with_name(p)) for p in
                         ("alfworld_short_extension.py", "alfworld_placement_test.py", "alfworld_stage_pairs.py")}}
    path = args.out / "extension-freeze.json"
    if not path.exists():
        torch.set_num_threads(1)
        dev = read(source_data)
        original_model = original["model"]
        blind = fit(mask([row for row in dev["train"] if row["features"] is not None]),
                    original_model["width"], original_model["penalty"])
        validation = mask(dev["validation"])
        gate = select(validation, predict(validation, blind))
        save(path, {"protocol": protocol, "model": original_model, "rules": original["rules"],
                    "without_proposal": blind, "ablation_rules": gate})
    release = read(path)
    if release["protocol"] != protocol:
        raise ValueError("locked extension inputs were changed")
    return release


def score_one(args, spec, actor, bank, cap):
    base = cached(receipt(args.out, spec), lambda: run(args, spec, actor, cap))
    a, b, x, target = base["metrics"], base["metrics"], None, None
    found = choose_stage(actor, base["trace"]["steps"], bank, "grounded-conflict")
    if found is not None:
        anchor, card, command = found
        x = stage_features(anchor, card, command, cap)
        target = {"step": len(anchor["prefix"]) + 1, "card": card["id"], "original": anchor["draft"]["action"],
                  "advice": command}
        save(receipt(args.out, spec, "pre-action"), {**target, "features": x})
        reminder = rendered(card["text"], command, True)
        after = cached(receipt(args.out, spec, "remind"), lambda: run(args, spec, actor, cap, anchor, command, reminder))
        if after["extra_draft"] is None:
            x = None
        else:
            index = len(anchor["prefix"])
            if (after["extra_draft"] != anchor["draft"] or after["trace"]["steps"][:index] != anchor["prefix"]
                    or after["trace"]["steps"][index]["state"] != anchor["state"]):
                raise ValueError("short extension fork used a different checkpoint")
            b = after["metrics"]
    return {"game": spec["id"], "template": spec["template"], "setting": SETTING, "features": x, "match": target,
            "silent": a, "remind": b, "extra": b["tokens"] - a["tokens"],
            "R": int(a["success"] < b["success"]), "H": int(a["success"] > b["success"])}


def confirm_live_budget(args, actor, transfer, root, manifest, bank):
    if (args.out / "budget-audit.json").exists():
        return
    audited = []
    cap = transfer["token_budgets"][SETTING]
    for s in manifest["splits"]["development"][:4]:
        old = read(receipt(root, s))
        live = run(args, s, actor, cap)
        want = token_prefix(old, cap)
        old_rows = old["trace"]["steps"][:want["steps"]]
        steps = live["trace"]["steps"]
        if (any(live["metrics"][k] != want[k] for k in ("success", "tokens", "actor_calls", "steps"))
                or [r["execution"]["state"] for r in steps] != [r["execution"]["state"] for r in old_rows]):
            raise ValueError("live token cutoff did not reproduce the archived silent prefix")
        audited.append({"game": s["id"], "steps": len(steps), "tokens": want["tokens"]})
    candidates = {s["id"]: s for group in ("valid_seen", "valid_unseen") for s in manifest["splits"][group]}
    aimed = []
    for path in sorted(args.development.glob("valid_*/*/scored.json")):
        source = read(path)
        opportunity = source["match"]
        row = next(r for r in source["rows"] if r["setting"] == SETTING)
        if not opportunity or row["features"] is None:
            continue
        spec = candidates[row["game"]]
        old = read(receipt(root, spec))
        after = read(receipt(args.development, spec, "remind"))
        anchor = anchor_at(old["trace"]["steps"], opportunity["step"] - 1)
        card = next(c for c in bank if c["id"] == opportunity["card"])
        cmd = opportunity["command"]
        live = run(args, spec, actor, cap, anchor, cmd, rendered(card["text"], cmd, True))
        want = score_tokens(old, opportunity, {"episodic": after}, cap)["episodic"]
        if any(live["metrics"][k] != want[k] for k in ("success", "tokens", "actor_calls", "steps")):
            raise ValueError("live token cutoff did not reproduce the archived reminder prefix")
        aimed.append({"game": spec["id"], "steps": want["steps"], "tokens": want["tokens"]})
        if len(aimed) == 3:
            break
    if len(aimed) < 3:
        raise ValueError("insufficient archived reminder trajectories to validate the live cutoff")
    save(args.out / "budget-audit.json", {"silent": audited, "remind": aimed})


def evaluate(args):
    release = lock(args)
    seal = read(args.out / "sealed-test.json")
    args.selection = "grounded-conflict-live"
    transfer, base, manifest, _, frozen, bank = configuration(args)
    if seal["budget"] != transfer["token_budgets"][SETTING]:
        raise ValueError("changed sealed token budget")
    actor = BudgetActor(frozen.actor, frozen.ranker, frozen.top, frozen.repeat_aware, frozen.examples, frozen.indexed)
    confirm_live_budget(args, actor, transfer, base, manifest, bank)
    rows = []
    new = 0
    specs = seal["splits"]["test"]
    for i, spec in enumerate(specs, 1):
        path = receipt(args.out, spec, "scored")
        exists = path.exists()
        rows.append(cached(path, lambda: score_one(args, spec, actor, bank, seal["budget"])))
        if not exists:
            new += 1
            save(args.out / "progress.json", {"completed": i, "total": len(specs)})
            print("SHORT_EXTENSION", i, len(specs), spec["id"], flush=True)
        if new == 90 and i < len(specs):
            os.execv(sys.executable, args.restart)
    prior = [r for p in args.previous.glob("stage-test/*/scored.json") for r in read(p)["rows"] if r["setting"] == SETTING]
    if len(prior) != 167 or {r["game"] for r in prior} & {r["game"] for r in rows}:
        raise ValueError("wrong previous short test cohort")
    independent = summarize(rows, release, mask)
    pooled = summarize(prior + rows, release, mask)
    save(args.out / "test-report.json", {"games": len(rows), "templates": len({r["template"] for r in rows}),
                                        "budget": seal["budget"], "independent_extension": independent,
                                        "pooled_games": len(prior) + len(rows), "pooled": pooled})
    print("SHORT_EXTENSION_COMPLETE", independent["demonstrated_goal"], pooled["demonstrated_goal"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "test"))
    for name in ("out", "study", "refit", "data", "config", "weights", "development", "previous"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.restart = [sys.executable, "-u", "-m", __spec__.name, *sys.argv[1:]]
    (precommit if args.phase == "register" else evaluate)(args)


if __name__ == "__main__":
    main()
