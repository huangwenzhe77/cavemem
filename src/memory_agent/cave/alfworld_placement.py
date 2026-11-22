"""Verify source placement memories and collect action-conditioned target pairs."""

import argparse
import contextlib
import random
from collections import Counter
from pathlib import Path

import numpy as np

from memory_agent.cave.alfworld_env import MAX_STEPS, sha_file
from memory_agent.cave.alfworld_ranker import RankedActor, entities, words
from memory_agent.cave.alfworld_stage_pairs import configuration, short_arm, summarize
from memory_agent.cave.alfworld_study_pairs import anchor_at, cached, rendered
from memory_agent.cave.alfworld_study_rollout import receipt, rollout
from memory_agent.cave.alfworld_transfer_study import draft_tokens, score_tokens
from memory_agent.cave.records import read, save


SEED = 262739
MODIFIERS = {"cool": "cool", "hot": "heat", "heat": "heat", "clean": "clean"}
FAMILIES = (
    "pick_and_place_simple", "pick_clean_then_place_in_recep", "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep", "pick_two_obj_and_place",
)


class PlacementActor(RankedActor):
    required = None

    def select_index(self, episode, obs, choices, reminder, original, start):
        if reminder and original is not None and self.required is not None:
            if self.required not in episode.commands:
                raise ValueError("source-mapped action is no longer legal")
            if self.required not in choices:
                choices = [*choices, self.required]
        return super().select_index(episode, obs, choices, reminder, original, start)

    @contextlib.contextmanager
    def include(self, action):
        self.required = action
        try:
            yield
        finally:
            self.required = None


def placements(rows, vocabulary, *, disagreement=False):
    """Yield verified progress opportunities using only the prefix and current public menu."""
    if not rows:
        return
    objective = words(rows[0]["goal"])
    roles = list(dict.fromkeys(w for w in objective if w in vocabulary))
    if len(roles) < 2:
        return
    item, destination = roles[:2]
    modifiers = {MODIFIERS[w] for w in objective if w in MODIFIERS}
    prepared, holding, placed = set(), set(), set()
    for i, row in enumerate(rows[:MAX_STEPS - 1]):
        original = row["draft"]["action"]
        options = []
        for command in row["public_menu"]:
            if not command.startswith(("move ", "put ")):
                continue
            args = entities(command)
            if (len(args) == 2 and args[0][0] == item and args[1][0] == destination
                    and args[0] in holding and all((args[0], act) in prepared for act in modifiers)):
                if not disagreement or command != original:
                    options.append(command)
        if options:
            # Preserve agreement when the Actor already chose a valid placement.
            chosen = original if original in options else min(options)
            yield anchor_at(rows, i), chosen, len(placed), len(options), roles
        execution = row["execution"]
        command = execution["action"]
        a = entities(command)
        feedback = words(execution["observation"])
        if not a or len(feedback) < 2 or feedback[0] != "you":
            continue
        verb = command.split()[0]
        if verb == "take" and (feedback[1] == "take" or feedback[:3] == ["you", "pick", "up"]):
            holding.add(a[0])
            placed.discard(a[0])
        elif verb in {"move", "put"} and feedback[1].startswith(("move", "put")):
            holding.discard(a[0])
            if len(a) == 2 and a[0][0] == item and a[1][0] == destination:
                placed.add(a[0])
        elif verb in {"cool", "heat", "clean"} and feedback[1].startswith(verb):
            prepared.add((a[0], verb))


def notice(action):
    return rendered("The requested item is ready and in hand. Its goal receptacle is within reach. Complete the placement now.", action, True)


def branch(args, spec, actor, baseline, hit):
    anchor, action = hit[:2]
    p = receipt(args.out, spec, "first-redraft")
    first_cap = len(anchor["prefix"]) + 2
    with actor.include(action):
        first = cached(p, lambda: rollout(args, spec, actor, anchor=anchor, reminder=notice(action), cap=first_cap))
        treated = cached(receipt(args.out, spec, "remind"),
                         lambda: short_arm(args, spec, actor, baseline, anchor, notice(action), first))
    return treated, first


def source(args, actor, base, manifest):
    archive = args.supplement
    extra = read(archive / "manifest.json")
    extras = set(extra["heads_splits"]["train"])
    input_specs = [(spec, base) for spec in manifest["splits"]["historical"]]
    input_specs += [(spec, archive) for spec in extra["splits"]["development"] if spec["id"] in extras]
    vocabulary = actor.ranker.vocabulary
    certified = []
    for spec, root in input_specs:
        if spec["family"] not in FAMILIES or sum(x["family"] == spec["family"] for x in certified) >= 2:
            continue
        natural = read(receipt(root, spec))
        if natural["spec"] != spec or natural["trace"]["success"]:
            continue
        candidates = list(placements(natural["trace"]["steps"], vocabulary, disagreement=True))
        if not candidates:
            continue
        # A full one-step correction is cheapest to validate after earlier placements on two-object tasks.
        anchor, action, prior, slots, roles = max(candidates, key=lambda x: (x[2], -len(x[0]["prefix"])))
        hit = (anchor, action, prior, slots, roles)
        direct_path = receipt(args.out, spec, "direct")
        direct = cached(direct_path, lambda: rollout(args, spec, actor, anchor=anchor, correction=action))
        if not direct["metrics"]["success"]:
            print("SOURCE_DIRECT_FAILED", spec["id"], flush=True)
            continue
        reminder, _ = branch(args, spec, actor, natural, hit)
        if not reminder["metrics"]["success"]:
            print("SOURCE_TEXT_FAILED", spec["id"], flush=True)
            continue
        recheck_path = receipt(args.out, spec, "remind-recheck")
        with actor.include(action):
            recheck = cached(recheck_path, lambda: rollout(args, spec, actor, anchor=anchor, reminder=notice(action)))
        if not recheck["metrics"]["success"]:
            print("SOURCE_TEXT_NOT_REPRODUCED", spec["id"], flush=True)
            continue
        evidence = {
            "source": spec["id"], "family": spec["family"], "template": spec["template"],
            "anchor_step": len(anchor["prefix"]) + 1, "anchor_state": anchor["state"],
            "original": anchor["draft"]["action"], "correction": action, "roles": roles, "prior_placements": prior,
            "natural_sha256": sha_file(receipt(root, spec)), "direct_sha256": sha_file(direct_path),
            "remind_sha256": sha_file(receipt(args.out, spec, "remind")),
            "recheck_sha256": sha_file(recheck_path),
            "remind_utility_gain": min(run["metrics"]["utility"] - natural["metrics"]["utility"]
                                       for run in (reminder, recheck)),
        }
        if evidence["remind_utility_gain"] > 0:
            certified.append(evidence)
            print("SOURCE_CERTIFIED", spec["id"], spec["family"], flush=True)
    save(args.out / "placement-bank.json", {"cards": certified, "source_cohort": len(input_specs)})
    if not certified:
        raise ValueError("no natural source failure repaired by an actual reminder")


def features(hit, family, budget, evidence):
    checkpoint, action, completed, count, roles = hit
    draft = checkpoint["draft"]
    original = draft["action"]
    logits = np.asarray(draft["choice_logits"], dtype=float)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    prefix = checkpoint["prefix"]
    spent = sum(draft_tokens(row["draft"]) for row in [*prefix, checkpoint])
    return [
        int(action == original), int(action not in draft["allowed"]), float(p.max()),
        float(-p @ np.log(np.maximum(p, 1e-12))), len(prefix), count, completed,
        len(draft["allowed"]), len(checkpoint["public_menu"]), budget or 0, max(0, budget - spent) if budget else 0,
        len(evidence), sum(r["draft"]["action"] == original for r in prefix[-8:]),
        int(original.startswith(("move ", "put "))),
        int(any(name == roles[0] for name, _ in entities(original))),
        int(any(name == roles[1] for name, _ in entities(original))),
        *(int(family == name) for name in FAMILIES),
    ]


def evaluate_game(args, spec, actor, natural, bank, budgets):
    support = [r for r in bank if r["family"] == spec["family"]]
    candidate = next(placements(natural["trace"]["steps"], actor.ranker.vocabulary), None) if support else None
    result = []
    match, treatment = None, natural
    if candidate:
        anchor, action = candidate[:2]
        matched_step = len(anchor["prefix"]) + 1
        match = {"step": matched_step, "original": anchor["draft"]["action"], "proposed": action}
        treatment, _ = branch(args, spec, actor, natural, candidate)
        if (treatment["extra_draft"] != anchor["draft"]
                or treatment["trace"]["steps"][matched_step - 1]["state"] != anchor["state"]):
            raise ValueError("reminder branch used a different pre-action checkpoint")
    for setting, token_cap in [("calls", None), *budgets.items()]:
        s, r, active = natural["metrics"], treatment["metrics"], match
        if token_cap is not None:
            paid = score_tokens(natural, match, {"episodic": treatment} if match else {}, token_cap)
            s, r, active = paid["silent"], paid["episodic"], paid["match"]
        result.append({"game": spec["id"], "template": spec["template"], "family": spec["family"], "setting": setting,
                       "features": features(candidate, spec["family"], token_cap, support) if active else None,
                       "silent": s, "remind": r, "R": int(r["success"] > s["success"]),
                       "H": int(s["success"] > r["success"]), "extra": r["tokens"] - s["tokens"],
                       "D": r["utility"] - s["utility"], "match": active})
    return result


def collect(args, actor, base, manifest, cohort, budgets):
    bank_path = args.source / "placement-bank.json"
    cards = read(bank_path)["cards"]
    if not cards:
        raise ValueError("cannot train with an empty verified placement bank")
    lock = args.out / "placement-bank.sha.json"
    checksum = {"sha256": sha_file(bank_path)}
    if lock.exists() and read(lock) != checksum:
        raise ValueError("the placement memory bank changed")
    save(lock, checksum)
    specs = {s["id"]: s for name in ("valid_seen", "valid_unseen") for s in manifest["splits"][name]}
    agenda = [(k, ident) for k in ("train", "validation") for ident in cohort[k]]
    random.Random(SEED).shuffle(agenda)
    rows = {"train": [], "validation": []}
    for i, (split, ident) in enumerate(agenda, 1):
        spec = specs[ident]
        natural = read(receipt(base, spec))
        data = cached(receipt(args.out, spec, "scored"),
                      lambda: evaluate_game(args, spec, actor, natural, cards, budgets))
        rows[split].extend(data)
        save(args.out / "progress.json", {"completed": i, "total": len(agenda), "metrics": {k: summarize(r) for k, r in rows.items()}})
        print("PLACEMENT_PROGRESS", i, len(agenda), ident, flush=True)
    save(args.out / "paired-labels.json", rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("source", "collect"))
    for name in ("out", "refit", "study", "data", "config", "weights", "supplement", "source"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args()
    args.selection = "verified-goal-placement"
    transfer, base, manifest, cohort, frozen, _ = configuration(args)
    if not frozen.indexed:
        raise ValueError("placement advisor needs the frozen indexed candidate actor")
    actor = PlacementActor(frozen.actor, frozen.ranker, frozen.top, frozen.repeat_aware, frozen.examples, frozen.indexed)
    if args.phase == "source":
        return source(args, actor, base, manifest)
    collect(args, actor, base, manifest, cohort, transfer["token_budgets"])


if __name__ == "__main__":
    main()
