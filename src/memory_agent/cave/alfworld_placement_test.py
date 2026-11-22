"""Test the verified placement advisor and learned gate at a locked live token budget."""

import argparse
import contextlib
import io
import os
import random
import sys
from pathlib import Path

from memory_agent.cave.alfworld_env import Episode, MAX_STEPS, sha_file
from memory_agent.cave.alfworld_experiment import costs
from memory_agent.cave.alfworld_full_study import available
from memory_agent.cave.alfworld_gate_refit import policy_report
from memory_agent.cave.alfworld_gate_test import interval
from memory_agent.cave.alfworld_placement import PlacementActor, features, notice, placements
from memory_agent.cave.alfworld_stage_pairs import configuration
from memory_agent.cave.alfworld_stage_policy import decide, fit, predict, select
from memory_agent.cave.alfworld_study_pairs import cached
from memory_agent.cave.alfworld_study_rollout import receipt
from memory_agent.cave.alfworld_transfer_study import draft_tokens
from memory_agent.cave.records import read, save


SEED = 262740
SETTING = "p50"
ACTION_FIELDS = {0, 1, 2, 3, 12, 13, 14, 15}


class BudgetReached(Exception):
    pass


class BudgetedModel:
    def __init__(self, model, left):
        self.model = model
        self.device = model.device
        self.left = left

    def __call__(self, **params):
        # The frozen indexed actor emits exactly one token from a numbered shortlist.
        if params["input_ids"].shape[1] + 1 > self.left:
            raise BudgetReached
        return self.model(**params)


class BudgetActor(PlacementActor):
    left = None

    def select_index(self, episode, obs, choices, reminder, original, start):
        original_model = self.actor.model
        if self.left is not None:
            self.actor.model = BudgetedModel(original_model, self.left)
        try:
            return super().select_index(episode, obs, choices, reminder, original, start)
        finally:
            self.actor.model = original_model


def run(args, spec, actor, budget, anchor=None, suggestion=None, message=None):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        initial = Episode(args.data, args.config, spec)
        ep = initial
        if anchor is not None:
            try:
                ep = initial.restore([r["execution"] for r in anchor["prefix"]], anchor["state"])
            finally:
                initial.env.close()
    rows = list(anchor["prefix"]) if anchor else []
    first = len(rows)
    extra = None
    spent = sum(draft_tokens(r["draft"]) for r in rows)
    try:
        while not ep.terminal and len(rows) + int(extra is not None) < MAX_STEPS:
            revising = anchor is not None and len(rows) == first
            if revising and len(rows) + 1 >= MAX_STEPS:
                break
            original = anchor["draft"] if revising else None
            actor.left = budget - spent - (draft_tokens(original) if revising else 0)
            try:
                if revising:
                    with actor.include(suggestion):
                        draft = actor.propose(ep, message or notice(suggestion), original["action"])
                else:
                    draft = actor.propose(ep)
            except BudgetReached:
                break
            if revising:
                extra = original
                spent += draft_tokens(original)
            spent += draft_tokens(draft)
            row = {"observation": ep.observation, "goal": ep.goal, "state": ep.signature(),
                   "draft": draft, "public_menu": ep.commands, "execution": ep.advance(draft["action"])}
            rows.append(row)
        trace = {"success": int(ep.won), "steps": rows, "end_state": ep.signature()}
        bill = costs(trace, first, extra) if extra is not None else costs(trace)
        if bill["tokens"] != spent or spent > budget:
            raise ValueError("wrong live budget accounting")
        return {"spec": spec, "trace": trace, "metrics": bill, "extra_draft": extra}
    finally:
        actor.left = None
        ep.env.close()


def mask(rows):
    return [{**r, "features": [v if i not in ACTION_FIELDS else 0 for i, v in enumerate(r["features"])]}
            if r["features"] is not None else r for r in rows]


def precommit(args):
    path = args.out / "sealed-test.json"
    if path.exists():
        raise FileExistsError(path)
    transfer_file = args.study / "transfer-freeze.json"
    transfer = read(transfer_file)
    root = args.study.parent
    known = sorted(root.glob("cave-alfworld*/manifest.json"))
    rejected = [r for old in known for rows in read(old).get("splits", {}).values() for r in rows]
    templates = {r.get("template") for r in rejected}
    rejected += [r for old in root.glob("cave-alfworld*/sealed-test.json")
                 for rows in read(old).get("splits", {}).values() for r in rows]
    games = {r.get("game") for r in rejected}
    hashes = {r.get("sha256") for r in rejected}
    pool = [r for r in available(args.data, "train") if r["template"] not in templates
            and r["file"] not in games and r["file_sha256"] not in hashes]
    if len({r["template"] for r in pool}) < 100:
        raise ValueError("not enough held-out templates")
    rng = random.Random(SEED)
    rng.shuffle(pool)
    specs = [{"id": f"place-test-{i:03d}", "family": r["family"], "template": r["template"],
              "game": r["file"], "sha256": r["file_sha256"], "seed": rng.randrange(2**32)}
             for i, r in enumerate(pool, 1)]
    save(path, {"splits": {"test": specs}, "transfer_sha256": sha_file(transfer_file),
                "config_sha256": sha_file(args.config), "seed": SEED, "budget": transfer["token_budgets"][SETTING]})
    print("SEALED_GAMES", len(specs), len({r["template"] for r in specs}), flush=True)


def score_game(args, spec, actor, bank, budget):
    s = cached(receipt(args.out, spec), lambda: run(args, spec, actor, budget))
    support = [r for r in bank if r["family"] == spec["family"]]
    candidate = next(placements(s["trace"]["steps"], actor.ranker.vocabulary), None) if support else None
    r, x, chosen = s, None, None
    if candidate is not None:
        anchor, action = candidate[:2]
        x = features(candidate, spec["family"], budget, support)
        chosen = {"step": len(anchor["prefix"]) + 1, "state": anchor["state"],
                  "original": anchor["draft"]["action"], "advice": action}
        save(receipt(args.out, spec, "pre-action"), {**chosen, "features": x})
        path = receipt(args.out, spec, "remind")
        attempted = cached(path, lambda: run(args, spec, actor, budget, anchor, action))
        if attempted["extra_draft"] is None:
            x = None  # Ineligible because the reminder cannot fit under the same budget.
        else:
            index = len(anchor["prefix"])
            if (attempted["extra_draft"] != anchor["draft"]
                    or attempted["trace"]["steps"][:index] != anchor["prefix"]
                    or attempted["trace"]["steps"][index]["state"] != anchor["state"]):
                raise ValueError("placement fork departed from the wrong state")
            r = attempted
    return {"game": spec["id"], "template": spec["template"], "setting": SETTING, "features": x, "match": chosen,
            "silent": s["metrics"], "remind": r["metrics"], "extra": r["metrics"]["tokens"] - s["metrics"]["tokens"],
            "R": int(r["metrics"]["success"] > s["metrics"]["success"]),
            "H": int(r["metrics"]["success"] < s["metrics"]["success"])}


def summarize(rows, policy, masker=mask):
    mlp = decide(rows, predict(rows, policy["model"]), policy["rules"])
    masked = masker(rows)
    no_proposal = decide(masked, predict(masked, policy["without_proposal"]), policy["ablation_rules"])
    comparisons = {"silent": [False] * len(rows), "always_remind": [r["features"] is not None for r in rows],
                   "action_disagreement": [r["features"] is not None and not r["features"][0] for r in rows],
                   "without_proposal": no_proposal, "mlp": mlp}
    reports = {key: policy_report(rows, choice)[SETTING] for key, choice in comparisons.items()}
    effects = {}
    for key, choice in comparisons.items():
        if key != "mlp":
            diff = [int(a) - int(b) for a, b in zip(mlp, choice, strict=True)]
            effects[key] = {"success_gain": interval(rows, [d * (r["R"] - r["H"]) for d, r in zip(diff, rows, strict=True)]),
                            "token_saving": interval(rows, [-d * r["extra"] for d, r in zip(diff, rows, strict=True)])}
    gate = reports["mlp"]
    point = all(gate["wins"] >= reports[k]["wins"] and gate["tokens"] <= reports[k]["tokens"]
                for k in ("silent", "always_remind"))
    main = (point and gate["rescues"] > 0 and effects["silent"]["success_gain"]["ci95"][0] > 0
            and effects["always_remind"]["success_gain"]["ci95"][0] >= -.01
            and all(effects[k]["token_saving"]["ci95"][0] > 0 for k in ("silent", "always_remind")))
    return {"policies": reports, "effects": effects, "point_goal": point, "demonstrated_goal": main,
            "avoided_harms": sum(r["H"] for r, a in zip(rows, mlp, strict=True) if not a),
            "missed_rescues": sum(r["R"] for r, a in zip(rows, mlp, strict=True) if not a),
            "eligible": sum(r["features"] is not None for r in rows)}


def evaluate(args):
    seal_path = args.out / "sealed-test.json"
    seal = read(seal_path)
    model_path = args.development / "model.json"
    source_path = args.source / "placement-bank.json"
    labels = args.development / "paired-labels.json"
    trained = read(model_path)
    if (sha_file(labels) != trained["dataset_sha256"]
            or read(args.development / "placement-bank.sha.json")["sha256"] != sha_file(source_path)
            or seal["transfer_sha256"] != sha_file(args.study / "transfer-freeze.json")
            or seal["config_sha256"] != sha_file(args.config)):
        raise ValueError("sealed placement inputs changed")
    locked = args.out / "test-freeze.json"
    hashes = {"manifest": sha_file(seal_path), "training": sha_file(model_path),
              "bank": sha_file(source_path), "code": sha_file(Path(__file__))}
    if not locked.exists():
        import torch

        torch.set_num_threads(1)
        data = read(labels)
        model = trained["model"]
        comparator = fit(mask([r for r in data["train"] if r["features"] is not None]), model["width"], model["penalty"])
        validation = mask(data["validation"])
        ablation_rules = select(validation, predict(validation, comparator))
        save(locked, {"hashes": hashes, "model": model, "rules": trained["rules"],
                      "without_proposal": comparator, "ablation_rules": ablation_rules})
    policy = read(locked)
    if policy["hashes"] != hashes:
        raise ValueError("placement gate changed after release")
    args.selection = "verified-placement-live"
    _, _, _, _, frozen, _ = configuration(args)
    actor = BudgetActor(frozen.actor, frozen.ranker, frozen.top, frozen.repeat_aware, frozen.examples, frozen.indexed)
    bank = read(source_path)["cards"]
    rows = []
    new = 0
    for i, spec in enumerate(seal["splits"]["test"], 1):
        path = receipt(args.out, spec, "scored")
        exists = path.exists()
        rows.append(cached(path, lambda: score_game(args, spec, actor, bank, seal["budget"])))
        if not exists:
            new += 1
            save(args.out / "progress.json", {"completed": i, "total": len(seal["splits"]["test"])})
            print("PLACEMENT_TEST", i, len(seal["splits"]["test"]), spec["id"], flush=True)
        if new == 90 and i < len(seal["splits"]["test"]):
            os.execv(sys.executable, args.restart)
    result = summarize(rows, policy)
    save(args.out / "test-report.json", {**result, "budget": seal["budget"], "games": len(rows),
                                        "templates": len({r["template"] for r in rows}),
                                        "test_freeze_sha256": sha_file(locked)})
    print("PLACEMENT_TEST_COMPLETE", result["demonstrated_goal"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "test"))
    for name in ("out", "study", "refit", "data", "config", "weights", "source", "development"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.restart = [sys.executable, "-u", "-m", __spec__.name, *sys.argv[1:]]
    (precommit if args.phase == "register" else evaluate)(args)


if __name__ == "__main__":
    main()
