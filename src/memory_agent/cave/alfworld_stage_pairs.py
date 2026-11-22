"""Collect paired outcomes for one short reminder at a context-supported ALFWorld action."""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from memory_agent.cave.alfworld_env import MAX_STEPS, Actor, sha_file
from memory_agent.cave.alfworld_experiment import costs
from memory_agent.cave.alfworld_ranker import RankedActor, entities, examples, load, shape, words
from memory_agent.cave.alfworld_study_pairs import anchor_at, cached, context_applies, contextual_contract, rendered
from memory_agent.cave.alfworld_study_rollout import receipt, rollout
from memory_agent.cave.alfworld_transfer_study import goal_profile, score_tokens, similarity
from memory_agent.cave.records import read, save


def public_progress(actor, anchor, command, card):
    """Recognize explicitly evidenced task progress without using an episode verifier."""
    vocabulary = actor.ranker.vocabulary
    roles = list(dict.fromkeys(w for w in words(anchor["goal"]) if w in vocabulary))
    if not roles:
        return False
    original = card["contract"]
    state = contextual_contract(actor, anchor, command)
    if state["argument_roles"] != original["argument_roles"]:
        return False
    if any(required and not visible for required, visible in
           zip(original["arguments_visible"], state["arguments_visible"], strict=True)):
        return False
    held, prepared = set(), set()
    for step in anchor["prefix"]:
        action = step["execution"]["action"]
        feedback = words(step["execution"]["observation"])
        references = entities(action)
        if not references or len(feedback) < 2 or feedback[0] != "you":
            continue
        verb = action.split()[0]
        item = references[0]
        if verb == "take" and (feedback[1].startswith("take") or feedback[:3] == ["you", "pick", "up"]):
            held.add(item)
        elif verb in {"move", "put"} and feedback[1].startswith(("move", "put")):
            held.discard(item)
        elif verb in {"cool", "heat", "clean"} and feedback[1].startswith(verb):
            prepared.add((item, verb))
    action = command.split()[0]
    refs = entities(command)
    if action == "take":
        return refs[0][0] == roles[0] and refs[0] not in held and all(state["arguments_visible"])
    if action not in {"go", "open"} or len(roles) < 2 or refs[0][0] != roles[1]:
        return False
    required = {"cool": "cool", "hot": "heat", "heat": "heat", "clean": "clean"}
    modifiers = {required[word] for word in words(anchor["goal"]) if word in required}
    requested = [item for item in held if item[0] == roles[0] and all((item, verb) in prepared for verb in modifiers)]
    return len(requested) >= (2 if "two" in words(anchor["goal"]) else 1)


def choose_stage(actor, rows, cards, selection="stage"):
    """Scan in execution order and commit at the first context-supported candidate."""
    if not rows:
        return None
    vocabulary = actor.ranker.vocabulary
    intent, nouns = goal_profile(vocabulary, rows[0]["goal"])
    candidates = []
    for card in cards:
        card_intent, card_nouns = goal_profile(vocabulary, card["source_goal"])
        relevance = similarity(intent, card_intent)
        if relevance >= .5:
            candidates.append((card, relevance, similarity(nouns, card_nouns)))
    for i, row in enumerate(rows[1:MAX_STEPS - 1], 1):
        anchor = anchor_at(rows, i)
        commands = row["draft"]["ranked_candidates"]
        shapes = [shape(command) for command in commands]
        hits = []
        for card, relevance, lexical in candidates:
            for j, (command, kind) in enumerate(zip(commands, shapes, strict=True)):
                if kind != card["shape"] or (selection != "stage" and command == row["draft"]["action"]):
                    continue
                strict = context_applies(actor, anchor, command, card)
                grounded = selection == "grounded-conflict" and public_progress(actor, anchor, command, card)
                eligible = grounded if selection == "grounded-conflict" else strict
                if eligible:
                    priority = strict, relevance, lexical, card["utility"], -j, card["id"]
                    hits.append((priority, card, command))
        if hits:
            _, card, command = max(hits)
            return anchor, card, command
    return None


def stage_features(anchor, card, advice, budget):
    draft = anchor["draft"]
    prefix = anchor["prefix"]
    commands = draft["allowed"]
    index = commands.index(advice)
    probs = np.exp(np.asarray(draft["choice_logits"]) - max(draft["choice_logits"]))
    probs /= probs.sum()
    spent = sum(r["draft"]["input_tokens"] + r["draft"]["output_tokens"] for r in [*prefix, anchor])
    obs = anchor["observation"]
    last_obs = [r["observation"] for r in prefix[-4:]]
    prior = [r["execution"]["action"] for r in prefix]
    return [
        int(draft["action"] == advice), index, float(probs[index]), float(probs.max()),
        float(-probs @ np.log(np.maximum(probs, 1e-12))), len(commands),
        len(anchor["public_menu"]), len(prior), prior.count(advice), prior.count(draft["action"]),
        sum(shape(action) == shape(advice) for action in prior[-6:]),
        last_obs.count(obs), len(set(last_obs)), len(obs.split()), card["direct_remaining"],
        card["utility"], len(card["text"].split()), budget or 0, max(0, budget - spent) if budget else 0,
        int(advice.split()[0] == "go"), int(advice.split()[0] == "take"),
        int(advice.split()[0] == "close"), int(advice.split()[0] == "open"),
    ]


def short_arm(args, spec, actor, natural, anchor, text, first):
    """Reuse a deterministic silent suffix when the transient prompt leaves the action unchanged."""
    i = len(anchor["prefix"])
    old = natural["trace"]["steps"]
    first_step = first["trace"]["steps"][-1]
    if first_step["draft"]["action"] != anchor["draft"]["action"]:
        return {**rollout(args, spec, actor, anchor=anchor, reminder=text), "shared_suffix": False}
    if first_step["execution"]["state"] != old[i]["execution"]["state"]:
        raise ValueError("same action did not preserve the paired environment state")
    rows = [*old[:i], first_step, *old[i + 1:MAX_STEPS - 1]]
    trace = {"steps": rows, "success": int(natural["trace"]["success"] and len(rows) == len(old)),
             "end_state": rows[-1]["execution"]["state"]}
    return {"spec": spec, "trace": trace, "metrics": costs(trace, i, anchor["draft"]),
            "extra_draft": anchor["draft"], "shared_suffix": True}


def collect_one(args, spec, actor, bank, budgets, natural):
    if natural["spec"] != spec or natural["extra_draft"] is not None:
        raise ValueError(f"unexpected natural trajectory for {spec['id']}")
    chosen = choose_stage(actor, natural["trace"]["steps"], bank, args.selection)
    match, treatment = None, natural
    rows = []
    if chosen is not None:
        anchor, card, advice = chosen
        step = len(anchor["prefix"]) + 1
        match = {"step": step, "card": card["id"], "command": advice, "original": anchor["draft"]["action"],
                 "state": anchor["state"]}
        notice = rendered(card["text"], advice, True)
        initial_path = receipt(args.out, spec, "first-redraft")
        first = cached(initial_path, lambda: rollout(args, spec, actor, anchor=anchor, reminder=notice, cap=step + 1))
        arm_path = receipt(args.out, spec, "remind")
        treatment = cached(arm_path, lambda: short_arm(args, spec, actor, natural, anchor, notice, first))
        trace = treatment["trace"]["steps"]
        if (treatment["spec"] != spec or treatment["extra_draft"] != anchor["draft"]
                or trace[:step - 1] != anchor["prefix"] or trace[step - 1]["state"] != anchor["state"]):
            raise ValueError("counterfactual reminder did not begin at the archived checkpoint")
        # Confirm the suffix optimization once against an independent full run.
        verified = args.out / "deterministic-suffix-audit.json"
        if treatment["shared_suffix"] and not verified.exists():
            replay = rollout(args, spec, actor, anchor=anchor, reminder=notice)
            fields = ("success", "tokens", "actor_calls", "steps")
            if (any(replay["metrics"][key] != treatment["metrics"][key] for key in fields)
                    or [r["execution"]["state"] for r in replay["trace"]["steps"]]
                    != [r["execution"]["state"] for r in trace]):
                raise ValueError("deterministic suffix audit failed")
            save(verified, {"spec": spec["id"], "reference": sha_file(arm_path),
                            "states": len(trace), "tokens": replay["metrics"]["tokens"]})
        match.update(changed=trace[step - 1]["draft"]["action"] != match["original"],
                     evidence_sha256=sha_file(arm_path), first_redraft_sha256=sha_file(initial_path))
    for setting, budget in [("calls", None), *budgets.items()]:
        if budget is None:
            s, r, active = natural["metrics"], treatment["metrics"], match
        else:
            measured = score_tokens(natural, match, {"episodic": treatment} if match else {}, budget)
            s, r, active = measured["silent"], measured["episodic"], measured["match"]
        features = stage_features(anchor, card, advice, budget) if active else None
        rows.append({"game": spec["id"], "template": spec["template"], "family": spec["family"],
                     "setting": setting, "features": features, "silent": s, "remind": r,
                     "D": r["utility"] - s["utility"], "extra": r["tokens"] - s["tokens"],
                     "R": int(r["success"] and not s["success"]),
                     "H": int(s["success"] and not r["success"])})
    return {"match": match, "rows": rows}


def summarize(rows):
    by_budget = defaultdict(Counter)
    for r in rows:
        by_budget[r["setting"]].update(games=1, matched=int(r["features"] is not None),
                                       silent_wins=r["silent"]["success"], remind_wins=r["remind"]["success"],
                                       rescues=r["R"], harms=r["H"], extra_tokens=r["extra"])
    return dict(by_budget)


def configuration(args):
    transfer_file = args.study / "transfer-freeze.json"
    transfer = read(transfer_file)
    base = Path(transfer["baseline"])
    manifest_file = base / "manifest.json"
    actor_file = base / "actor.json"
    bank_file = args.study / "transfer-bank.json"
    config = read(actor_file)
    manifest = read(manifest_file)
    if (sha_file(manifest_file) != transfer["manifest_sha256"] or sha_file(actor_file) != transfer["actor_sha256"]
            or sha_file(bank_file) != transfer["bank_sha256"] or sha_file(base / "ranker.pt") != config["ranker"]
            or sha_file(args.config) != manifest["config_sha256"] or str(args.weights.resolve()) != config["actor"]):
        raise ValueError("source bank or actor no longer matches the archived study")
    cohort = read(args.refit / "cohort.json")
    if sha_file(transfer_file) != cohort["transfer_sha256"]:
        raise ValueError("incorrect training archive")
    seal = {"selection": args.selection, "treatment": "one short reminder",
            "transfer_sha256": sha_file(transfer_file), "cohort_sha256": sha_file(args.refit / "cohort.json"),
            "code_sha256": sha_file(Path(__file__))}
    path = args.out / "collector-freeze.json"
    if path.is_file() and read(path) != seal:
        raise ValueError("cannot resume with a different stage or treatment")
    save(path, seal)
    ranker = load(base)
    actor = RankedActor(Actor(args.weights), ranker, config["top"], config.get("repeat_aware", False),
                        examples(base, ranker) if config.get("examples") else (), config["indexed"])
    return transfer, base, manifest, cohort, actor, read(bank_file)["cards"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "refit", "study", "data", "config", "weights"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--selection", choices=("stage", "conflict", "grounded-conflict"), default="stage")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    transfer, base, manifest, cohort, actor, bank = configuration(args)
    specs = {s["id"]: s for k in ("valid_seen", "valid_unseen") for s in manifest["splits"][k]}
    rows = {"train": [], "validation": []}
    train, val = list(cohort["train"]), list(cohort["validation"])
    rng = random.Random(262736)
    rng.shuffle(train)
    rng.shuffle(val)
    queue = []
    for i, ident in enumerate(train):
        queue.append(("train", ident))
        if i % 4 == 0 and i // 4 < len(val):
            queue.append(("validation", val[i // 4]))
    if args.limit is not None:
        queue = queue[:args.limit]
    for i, (split, ident) in enumerate(queue, 1):
        spec = specs[ident]
        original = read(receipt(base, spec))
        evidence = cached(receipt(args.out, spec, "scored"),
                          lambda: collect_one(args, spec, actor, bank, transfer["token_budgets"], original))
        rows[split].extend(evidence["rows"])
        progress = {group: summarize(data) for group, data in rows.items()}
        save(args.out / "progress.json", {"completed": i, "goal": len(queue), "metrics": progress})
        print("STAGE_PROGRESS", i, len(queue), ident, split, flush=True)
    save(args.out / "paired-labels.json", rows)


if __name__ == "__main__":
    main()
