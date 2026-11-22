"""Measure transfer of source-verified episodic reminders separately from redrafting."""

import argparse
from collections import Counter
from pathlib import Path
import re
from math import ceil
from time import monotonic
from types import SimpleNamespace

from .alfworld_env import Actor, MAX_STEPS, sha_file
from .alfworld_experiment import costs
from .alfworld_ranker import RankedActor, entities, examples, load, shape, words
from .alfworld_study_pairs import FUNCTION_WORDS, anchor_at, bank_checked, cached, context_applies, rendered
from .alfworld_study_rollout import receipt, rollout
from .records import digest, read, save


NEUTRAL = "Review the task, current evidence, and your original proposal before choosing the next action."


def mapped_action(source_action, choices):
    source_args = set(entities(source_action))
    source_names = {name for name, _ in source_args}
    candidates = [a for a in choices if shape(a) == shape(source_action)]
    return max(candidates, key=lambda a: (
        a == source_action, len(set(entities(a)) & source_args),
        len({name for name, _ in entities(a)} & source_names),
    ), default=None)


def episodic_note(card, choices, representation):
    action = card["source_action"]
    if representation != "trajectory":
        candidate = mapped_action(action, choices)
        return rendered(card["episode"], candidate, candidate is not None)
    return rendered(card["episode"], action, action in choices)


def contextual_advice(card):
    entries = card["source_transitions"]

    def advise(ep, choices):
        recent = [shape(row["action"]) for row in ep.history[-3:]]
        observed = set(words(shape(ep.observation))) - FUNCTION_WORDS
        best = None
        for entry in entries:
            action = mapped_action(entry["action"], choices)
            if action is None:
                continue
            aligned = sum(left == right for left, right in zip(
                reversed(recent), reversed(entry["recent_shapes"]),
            ))
            lexical = similarity(observed, set(entry["observation_words"]))
            if not aligned and lexical < .35:
                continue
            score = (entry["observation"] == ep.observation, aligned, lexical)
            if best is None or score > best[0]:
                best = (score, entry, action)
        if best is None:
            return card["trajectory"]
        _, entry, action = best
        note = (f"Source task: {card['source_goal']} A similar observation in the successful source "
                f"trajectory was: {entry['observation']} The next historical action was {entry['action']}.")
        return rendered(note, action, True)

    return advise


def continuing_advice(card, prefix_length, args):
    if args.exposure == "transient":
        return None
    if args.representation == "trajectory":
        return card["trajectory"]
    if args.representation == "contextual":
        return contextual_advice(card)
    keep_trajectory = args.representation == "sequence-trajectory"

    def advise(ep, choices):
        index = card["source_step"] + len(ep.history) - prefix_length
        commands = card["source_commands"]
        if index >= len(commands):
            return card["trajectory"] if keep_trajectory else None
        source_action = commands[index]
        candidate = mapped_action(source_action, choices)
        context = card["trajectory"] if keep_trajectory else f"Source task: {card['source_goal']}"
        note = (f"{context} At step {index + 1} of the successful trajectory the next historical "
                f"action was {source_action}. Compare the two situations.")
        return rendered(note, candidate, candidate is not None)
    return advise


def build_bank(args, actor, manifest):
    release_path = args.source / "bank.json"
    release = bank_checked(SimpleNamespace(out=args.source))
    specs = {s["id"]: s for s in manifest["splits"]["historical"]}
    source_outputs = []
    for card in release["cards"]:
        proof_path = args.source / card["evidence"]
        proof = read(proof_path)
        spec = specs[card["source"]]
        if proof["source"] != spec["id"] or not spec["game"].startswith("json_2.1.1/train/"):
            raise ValueError("memory source is outside the registered history")
        if any([row[a]["success"] for a in ("silent", "direct", "remind")] != [0, 1, 1]
               for row in proof["metrics"]):
            raise ValueError("source card has no successful correction proof")
        match = re.search(r"-(\d+)-step-(\d+)$", proof_path.parent.name)
        anchor = proof["anchor"]
        if not match or int(match[2]) != len(anchor["prefix"]) + 1:
            raise ValueError("could not identify the archived source cap and checkpoint")
        cap = int(match[1])
        action = proof["action"]
        variant = proof_path.stem.removeprefix("accepted-")
        direct_path = proof_path.with_name(f"direct-{variant}-1.json")
        direct = read(direct_path)
        if direct["metrics"] != proof["metrics"][0]["direct"] or direct["spec"] != spec:
            raise ValueError("source trajectory differs from its admitted evidence")
        steps = direct["trace"]["steps"]
        commands = [row["execution"]["action"] for row in steps]
        source_step = len(anchor["prefix"])
        if commands[source_step] != action:
            raise ValueError("verified correction differs from the recorded source action")
        trajectory = f"Source task: {anchor['goal']} Successful source trajectory: {' → '.join(commands)}."
        episode = (f"{trajectory} Source observation at the recorded checkpoint: {anchor['observation']} "
                   f"The next source action that led to success was {action}. Source lesson: {card['text']}")
        candidate = {
            **card, "episode": episode, "trajectory": trajectory, "source_action": action, "source_goal": anchor["goal"],
            "source_commands": commands, "source_step": source_step,
            "source_cap": cap, "source_proof_sha256": sha_file(proof_path),
            "direct_sha256": sha_file(direct_path),
        }
        if args.representation == "contextual":
            candidate["source_transitions"] = [
                {
                    "action": command, "observation": state["observation"],
                    "observation_words": sorted(set(words(shape(state["observation"]))) - FUNCTION_WORDS),
                    "recent_shapes": [shape(x) for x in commands[max(0, i - 3):i]],
                }
                for i, (command, state) in enumerate(zip(commands, steps, strict=True))
            ]
        wording = episodic_note(candidate, anchor["draft"]["allowed"], args.representation)
        ongoing = continuing_advice(candidate, source_step, args)
        holder = args.out / "source-replays" / digest([candidate, spec, args.exposure, args.representation])
        ballots = []
        for trial in range(2):
            raw = cached(holder / f"raw-{trial}.json", lambda: rollout(
                args, spec, actor, anchor=anchor, reminder=wording, cap=cap,
                continuation_reminder=ongoing,
            ))
            score = raw["metrics"]
            if not score["success"] or score["utility"] <= proof["metrics"][trial]["silent"]["utility"]:
                break
            ballots.append(score)
        print("SOURCE_TRANSFER", spec["id"], len(ballots), flush=True)
        if len(ballots) == 2:
            candidate["source_replays"] = str(holder.relative_to(args.out))
            source_outputs.append(candidate)
    save(args.out / "transfer-bank.json", {
        "cards": source_outputs, "construction_bank_sha256": sha_file(release_path), "exposure": args.exposure,
        "representation": args.representation,
        "admission": "both episodic replays win and improve over the earlier verified silent arm",
    })
    if not source_outputs:
        raise RuntimeError("no source-verified episodic reminders")


def goal_profile(vocabulary, goal):
    terms = set(words(goal))
    return terms - vocabulary - FUNCTION_WORDS, terms & vocabulary


def similarity(left, right):
    return len(left & right) / max(1, len(left | right))


def choose_transfer(actor, rows, cards):
    """Select the first available related move, without reading any later transition."""
    if not rows:
        return None
    vocabulary = actor.ranker.vocabulary
    intent, entities = goal_profile(vocabulary, rows[0]["goal"])
    pool = []
    for card in cards:
        source_intent, source_entities = goal_profile(vocabulary, card["source_goal"])
        relevance = similarity(intent, source_intent)
        if relevance >= 0.5:
            pool.append((card, relevance, similarity(entities, source_entities)))
    for index, row in enumerate(rows[:MAX_STEPS - 1]):
        if not pool:
            break
        candidates = []
        for card, relevance, entity_similarity in pool:
            allowed = row["draft"]["ranked_candidates"]
            analogous = next((a for a in allowed if shape(a) == card["shape"]), None)
            if analogous is None:
                continue
            candidates.append((relevance, entity_similarity, card["utility"], card["id"], card, analogous))
        if candidates:
            _, _, _, _, card, analogous = max(candidates, key=lambda hit: hit[:4])
            anchor = anchor_at(rows, index)
            return anchor, card, analogous
    return None


def order_arms(seed, game_id):
    arms = ("neutral", "episodic")
    return arms if int(digest([seed, game_id])[:2], 16) % 2 == 0 else arms[::-1]


def draft_tokens(draft):
    return draft["input_tokens"] + draft["output_tokens"]


def token_prefix(run, budget):
    trace = run["trace"]
    extra = run["extra_draft"]
    spent = draft_tokens(extra) if extra is not None else 0
    rows = []
    for row in trace["steps"]:
        spent += draft_tokens(row["draft"])
        if spent > budget:
            break
        rows.append(row)
    # A trace terminates as soon as TextWorld reports a win.
    prefix = {"success": int(trace["success"] and len(rows) == len(trace["steps"])), "steps": rows}
    return costs(prefix, 0, extra) if extra is not None else costs(prefix)


def score_tokens(natural, match, branches, budget):
    silent = token_prefix(natural, budget)
    row = {"match": None, "silent": silent, "neutral": silent, "episodic": silent}
    if not match:
        return row
    step = match["step"]
    spent = sum(draft_tokens(r["draft"]) for r in natural["trace"]["steps"][:step])
    reserve = max(draft_tokens(run["trace"]["steps"][step - 1]["draft"]) for run in branches.values())
    if spent + reserve > budget:
        return row
    # Both redraft arms use the same admission rule, then stop before the next unaffordable call.
    row["match"] = match
    row.update({arm: token_prefix(run, budget) for arm, run in branches.items()})
    return row


def compare(args, actor, spec, natural, cards, seed):
    start = monotonic()
    choice = choose_transfer(actor, natural["trace"]["steps"], cards)
    matching_seconds = monotonic() - start
    output = {"spec": spec["id"], "match": None, "matching_seconds": matching_seconds,
              "silent": natural["metrics"], "neutral": natural["metrics"], "episodic": natural["metrics"]}
    if choice is None:
        return output
    anchor, card, analogous = choice
    choices = anchor["draft"]["allowed"]
    wordings = {"neutral": NEUTRAL, "episodic": episodic_note(card, choices, args.representation)}
    ongoing = {
        "neutral": NEUTRAL if args.exposure == "persistent" else None,
        "episodic": continuing_advice(card, len(anchor["prefix"]), args),
    }
    target = receipt(args.out, spec, "matched").parent
    output["match"] = {
        "step": len(anchor["prefix"]) + 1, "card": card["id"], "source": card["source"],
        "state": anchor["state"], "evidence": {},
        "strict_contract": context_applies(actor, anchor, analogous, card),
        "original": anchor["draft"]["action"], "analogous": analogous,
        "source_action_legal_in_shortlist": card["source_action"] in choices,
    }
    for arm in order_arms(seed, spec["id"]):
        treatment = [args.exposure, args.representation, card["id"]] if arm == "episodic" else [args.exposure]
        stem = arm + "-" + digest([spec, anchor["state"], wordings[arm], treatment])[:16]
        result = cached(target / f"{stem}.json", lambda: rollout(
            args, spec, actor, anchor=anchor, reminder=wordings[arm],
            continuation_reminder=ongoing.get(arm),
        ))
        output[arm] = result["metrics"]
        output["match"]["evidence"][arm] = {"file": f"{stem}.json", "sha256": sha_file(target / f"{stem}.json")}
        output["match"][arm + "_first_action"] = result["trace"]["steps"][len(anchor["prefix"])]["draft"]["action"]
    return output


def summary(records):
    counts = Counter()
    for row in records:
        s, n, m = (row[a] for a in ("silent", "neutral", "episodic"))
        dt, seconds = m["tokens"] - s["tokens"], m["seconds"] - s["seconds"]
        counts.update(
            tasks=1, matches=int(row["match"] is not None),
            silent_wins=s["success"], neutral_wins=n["success"], episodic_wins=m["success"],
            covered_silent_wins=int(bool(row["match"]) and s["success"]),
            harmed=int(s["success"] and not m["success"]),
            harmed_neutral=int(s["success"] and not n["success"]),
            harmed_by_content=int(s["success"] and n["success"] and not m["success"]),
            rescued=int(not s["success"] and m["success"]),
            harmed_with_more_cost=int(s["success"] and not m["success"] and dt > 0 and seconds > 0),
            dual_wins_more_cost=int(s["success"] and m["success"] and dt > 0 and seconds > 0),
            delta_tokens=dt, delta_seconds=seconds,
        )
    return dict(counts)


def audit_pair(args, spec, natural, paired, cards):
    if paired["spec"] != spec["id"] or paired["silent"] != natural["metrics"]:
        raise ValueError(f"silent evidence mismatch: {spec['id']}")
    match = paired["match"]
    if not match:
        if any(paired[arm] != paired["silent"] for arm in ("neutral", "episodic")):
            raise ValueError(f"unmatched game has an intervention: {spec['id']}")
        return {}
    if match["card"] not in cards:
        raise ValueError(f"unknown card: {spec['id']}")
    step = match["step"] - 1
    original = natural["trace"]["steps"][step]
    branches = {}
    for arm in ("neutral", "episodic"):
        evidence = match["evidence"][arm]
        path = receipt(args.out, spec).parent / evidence["file"]
        result = read(path)
        rows = result["trace"]["steps"]
        if (sha_file(path) != evidence["sha256"] or result["spec"] != spec
                or result["metrics"] != paired[arm] or result["extra_draft"] != original["draft"]
                or rows[:step] != natural["trace"]["steps"][:step]
                or rows[step]["state"] != match["state"] or original["state"] != match["state"]
                or not 0 < paired[arm]["actor_calls"] <= MAX_STEPS):
            raise ValueError(f"unverified {arm} branch: {spec['id']}")
        branches[arm] = result
    return branches


def run_pairs(args, manifest, actor, *, development):
    bank_path = args.out / "transfer-bank.json"
    bank = read(bank_path)
    cards = bank["cards"]
    if not cards or args.exposure != bank["exposure"] or args.representation != bank["representation"]:
        raise ValueError("transfer bank is empty or has a different exposure protocol")
    natural_development = [read(receipt(args.baseline, spec)) for spec in manifest["splits"]["development"]]
    winning_tokens = sorted(r["metrics"]["tokens"] for r in natural_development if r["metrics"]["success"])
    if not winning_tokens:
        raise ValueError("no successful development trajectories for token-budget calibration")
    token_budgets = {
        f"p{p}": winning_tokens[ceil(p * len(winning_tokens) / 100) - 1] for p in (50, 75, 90)
    }
    protocol = {
        "bank_sha256": sha_file(bank_path), "actor_sha256": sha_file(args.baseline / "actor.json"),
        "manifest_sha256": sha_file(args.baseline / "manifest.json"), "cap": MAX_STEPS,
        "point": "first prefix matching source goal intent and a currently shortlisted action shape",
        "arms": ["silent", "neutral", "episodic"], "baseline": str(args.baseline.resolve()),
        "exposure": args.exposure,
        "representation": args.representation,
        "token_budgets": token_budgets,
        "code": {name: sha_file(Path(__file__).with_name(name)) for name in (
            "alfworld_transfer_study.py", "alfworld_study_rollout.py", "alfworld_ranker.py",
        )},
    }
    freeze = args.out / "transfer-freeze.json"
    if not development and read(freeze) != protocol:
        raise ValueError("transfer protocol changed after development")
    groups = ("development",) if development else ("valid_seen", "valid_unseen")
    result = {"calls": {}}
    result.update({label: {} for label in token_budgets})
    card_ids = {card["id"] for card in cards}
    for group in groups:
        for table in result.values():
            table[group] = []
        rows = result["calls"][group]
        for spec in manifest["splits"][group]:
            natural = read(receipt(args.baseline, spec))
            if natural["spec"] != spec or natural["extra_draft"] is not None:
                raise ValueError("unexpected silent baseline")
            row = cached(receipt(args.out, spec, "paired"), lambda: compare(
                args, actor, spec, natural, cards, manifest["seed"],
            ))
            branches = audit_pair(args, spec, natural, row, card_ids)
            rows.append(row)
            for label, budget in token_budgets.items():
                result[label][group].append(score_tokens(natural, row["match"], branches, budget))
            print("TRANSFER_PAIR", spec["id"], bool(row["match"]),
                  [row[a]["success"] for a in protocol["arms"]], flush=True)
            save(args.out / "progress.json", {name: summary(r) for name, r in result["calls"].items()})
    report = {}
    for setting, cohorts in result.items():
        report[setting] = {group: summary(rows) for group, rows in cohorts.items()}
        report[setting]["overall"] = summary([row for rows in cohorts.values() for row in rows])
    if not development:
        expected = {"valid_seen": 140, "valid_unseen": 134}
        coverage = {name: len({row["spec"] for row in rows}) for name, rows in result["calls"].items()}
        if coverage != expected or report["calls"]["overall"]["tasks"] != sum(expected.values()):
            raise ValueError("incomplete official benchmark")
    save(args.out / ("development.json" if development else "summary.json"), report)
    if development:
        if report["calls"]["overall"]["covered_silent_wins"] < 4:
            raise RuntimeError("source reminders cover too few successful development games")
        save(freeze, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["source", "development", "test"])
    parser.add_argument("--exposure", choices=["transient", "persistent"], default="persistent")
    parser.add_argument("--representation", choices=["trajectory", "sequence", "sequence-trajectory", "contextual"],
                        default="trajectory")
    for name in ("baseline", "source", "out", "data", "config", "weights"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.out.resolve() in (args.source.resolve(), args.baseline.resolve()):
        raise ValueError("use a separate output directory")
    if args.stage != "test" and (args.out / "transfer-freeze.json").exists():
        raise ValueError("the output is already frozen for test")
    mpath = args.baseline / "manifest.json"
    manifest = read(mpath)
    if sha_file(mpath) != sha_file(args.source / "manifest.json"):
        raise ValueError("source and baseline manifests differ")
    if sha_file(args.config) != manifest["config_sha256"]:
        raise ValueError("environment configuration changed")
    config = read(args.baseline / "actor.json")
    if str(args.weights.resolve()) != config["actor"] or sha_file(args.baseline / "ranker.pt") != config["ranker"]:
        raise ValueError("actor or ranker changed")
    if not config.get("indexed"):
        raise ValueError("token budgets require the one-token indexed actor")
    ranker = load(args.baseline)
    actor = RankedActor(Actor(args.weights), ranker, config["top"], config.get("repeat_aware", False),
                        examples(args.baseline, ranker) if config.get("examples") else (),
                        config.get("indexed", False))
    if args.stage == "source":
        build_bank(args, actor, manifest)
    else:
        run_pairs(args, manifest, actor, development=args.stage == "development")


if __name__ == "__main__":
    main()
