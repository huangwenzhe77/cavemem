"""Staged ALFWorld pilot for source-verified reminders and shared call budgets."""
import argparse
import os
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from .alfworld import snapshot
from .alfworld_env import Actor, Episode, sha_file, visible_context
from .alfworld_experiment import costs
from .alfworld_memory import paired_contrast
from .models import TeacherCallError, teacher_request
from .records import digest, read, save


FAMILIES = ("pick_and_place_simple", "pick_clean_then_place_in_recep")
SPLITS = {"test": ("valid_unseen", 6), "development": ("train", 4), "source": ("train", 5)}
TEACHER = """Propose one short conditional memory to improve the failed source decision supplied as JSON.
The candidate legal correction is already supplied and separately executed. Explain the next-action logic
in general terms. Use public observations. Avoid giving an answer for a specific episode, hidden state,
or unvalidated multi-step route. Return JSON with a single string field reminder, at most 70 words."""


def register(args):
    rng = random.Random(args.seed)
    previously = args.out.parent / "cave-alfworld-01" / "manifest.json"
    used = ({x["template"] for rows in read(previously)["splits"].values() for x in rows}
            if previously.exists() else set())
    splits = {}
    for split, (bucket, quota) in SPLITS.items():
        selected = []
        for family in FAMILIES:
            pool = sorted((args.data / "json_2.1.1" / bucket).glob(f"{family}-*/*/game.tw-pddl"))
            rng.shuffle(pool)
            count = 0
            for path in pool:
                template = path.parent.parent.name.rsplit("-", 1)[0]
                if template in used or "Sliced" in template or not read(path)["solvable"]:
                    continue
                selected.append({"id": f"{split}-{len(selected)+1:02d}", "family": family,
                                 "template": template, "game": path.relative_to(args.data).as_posix(),
                                 "sha256": sha_file(path), "seed": rng.randrange(2**32)})
                used.add(template)
                count += 1
                if count == quota:
                    break
            if count != quota:
                raise ValueError(f"missing quota for {split}/{family}")
        splits[split] = selected
    save(args.out / "manifest.json", {"splits": splits, "seed": args.seed, "calls": args.calls,
                                     "config": sha_file(args.config), "protocol": "public-ALFWorld-harm-v1"})


def register_replication(args):
    parent = args.source_run
    original = read(parent/"manifest.json")
    bank_name = "freeze-general.json" if args.general else "freeze-greedy.json"
    bank = read(parent/bank_name)
    if not bank["bank"]:
        raise ValueError("replication needs a previously verified frozen bank")
    used = {x["template"] for rows in original["splits"].values() for x in rows}
    older = parent.parent/"cave-alfworld-01"/"manifest.json"
    if older.exists():
        used.update(x["template"] for rows in read(older)["splits"].values() for x in rows)
    if args.general:
        earlier = parent.parent/"cave-alfworld-harm-02"/"manifest.json"
        used.update(x["template"] for rows in read(earlier)["splits"].values() for x in rows)
    offset = 151 if args.general else 71
    rng = random.Random(original["seed"]+offset)
    bucket = "train" if args.general else "valid_seen"
    data = args.data/"json_2.1.1"/bucket
    family = FAMILIES[1]
    pool = sorted(data.glob(f"{family}-*/*/game.tw-pddl"))
    rng.shuffle(pool)
    selected = []
    for path in pool:
        template = path.parent.parent.name.rsplit("-", 1)[0]
        if template in used or "Sliced" in template or not read(path)["solvable"]:
            continue
        selected.append({"id": f"test-{len(selected)+1:02d}", "family":family,
                         "template":template,"game":path.relative_to(args.data).as_posix(),
                         "sha256":sha_file(path),"seed":rng.randrange(2**32)})
        used.add(template)
        if args.general and len(selected) >= 18:
            break
    save(args.out/"manifest.json", {"splits":{"test":selected}, "seed":original["seed"]+offset,
                                    "calls":[50],"local_budgets":[3,4,6],"primary_local_budget":3,
                                    "source":str(parent.resolve()),"source_digest":digest(bank),
                                    "source_bank":bank_name, "dataset_split":bucket, "config":sha_file(args.config),
                                    "protocol":"fixed remaining actor calls at first applicable public checkpoint"})
    print("REGISTERED REPLICATION",len(selected),flush=True)


def public_choices(ep):
    x = visible_context(ep)
    wanted = x["target"] or ""
    goal = ep.goal.lower().rstrip(".")
    dest = re.search(r" (?:in|on) (?:the |a |an )?(\w+)$", goal)
    place = dest[1] if dest else ""
    clean = "clean" in goal.split()
    action_history = [(r["action"], r["observation"].lower()) for r in ep.history]
    cleaned = any(a.startswith(f"clean {wanted} ") and "you clean" in o for a, o in action_history)
    touched = Counter(a for a, _ in action_history)
    cmds = ep.commands
    phase = x["phase"]
    take = [a for a in cmds if a.startswith(f"take {wanted} ")]
    drop = [a for a in cmds if a.startswith(("move ", "put "))]
    look_open = [a for a in cmds if a.startswith("open ") and not touched[a]]
    docks = (("sink", "bathtub") if clean and not cleaned else (place,))
    if phase == "holding_target":
        if clean and not cleaned:
            best = [a for a in cmds if a.startswith(f"clean {wanted} ")]
            kind = "treat"
        else:
            best = [a for a in cmds if a.startswith((f"move {wanted} ", f"put {wanted} "))
                    and re.search(rf"\b{re.escape(place)} \d+$", a)]
            kind = "deliver"
        if best:
            return kind, best, [a for a in cmds if a not in best][:2]
        best = [a for a in look_open if a.split()[1].startswith(docks)]
        best += [a for a in cmds if a.startswith("go to ") and a[6:].startswith(docks)]
        if best:
            few = min(touched[a] for a in best)
            return "route", [a for a in best if touched[a] == few][:5], []
        return "route_unknown", cmds[:5], []
    if phase == "holding_other" and drop:
        return "free_hand", drop[:5], []
    if take:
        return "take", take, [a for a in cmds if a not in take][:2]
    if look_open:
        return "open", look_open[:5], []
    walk = [a for a in cmds if a.startswith("go to ")]
    if walk:
        few = min(touched[a] for a in walk)
        return "search", [a for a in walk if touched[a] == few][:5], []
    return "unknown", cmds[:5], []


class Scaffold:
    def __init__(self, actor):
        self.actor = actor

    def propose(self, ep, reminder=None, original=None, **sampling):
        kind, preferred, nearby = public_choices(ep)
        commands = preferred + nearby
        note = ("\nPublic progress tracker. Current stage: " + kind +
                ". Commands that directly advance the stage: " + ", ".join(preferred) +
                ". Choose a progress command when available.")
        if original is not None and original not in commands:
            commands.append(original)
        view = SimpleNamespace(goal=ep.goal, history=ep.history, commands=commands,
                               observation=ep.observation+note)
        return self.actor.propose(view, reminder, original, **sampling)


def live(args, actor, spec, cap, path, *, anchor=None, replacement=None, reminder=None, seed=None):
    binding = digest([spec, cap, anchor, replacement, reminder, seed])
    if path.exists():
        old = read(path)
        if old.get("binding") != binding:
            raise ValueError(f"rollout inputs changed: {path}")
        return old
    ep = Episode(args.data, args.config, spec)
    if anchor is not None:
        original_ep = ep
        try:
            ep = ep.restore([r["execution"] for r in anchor["prefix"]], anchor["checkpoint"]["state"])
        finally:
            original_ep.env.close()
    rows = list(anchor["prefix"]) if anchor else []
    extra = None
    try:
        while not ep.terminal and len(rows) + int(extra is not None) < cap:
            ix = len(rows)
            sampling = ({} if seed is None else
                        {"seed": int(digest([seed, ix, "suffix"])[:12],16), "temperature": .7})
            kind, preferred, nearby = public_choices(ep)
            if anchor and ix == len(anchor["prefix"]):
                draft = anchor["checkpoint"]["draft"]
                if reminder is not None:
                    if len(rows)+1 >= cap:
                        raise ValueError("checkpoint cannot fund a reminder under the shared cap")
                    extra = draft
                    draft = actor.propose(ep, reminder, draft["action"], **sampling)
                elif replacement is not None:
                    draft = {**draft, "action": replacement}
            else:
                draft = actor.propose(ep, **sampling)
            cp = {**snapshot(ep, draft), "stage":kind, "preferred":preferred, "alternatives":nearby}
            cp["execution"] = ep.advance(draft["action"])
            rows.append(cp)
        trace = {"success": int(ep.won), "steps": rows, "end_state": ep.signature()}
        metrics = costs(trace) if extra is None else costs(trace, len(anchor["prefix"]), extra)
        out = {"spec":spec, "cap":cap, "trace":trace, "metrics":metrics, "binding":binding,
               "extra_draft":extra}
        save(path, out)
        return out
    finally:
        ep.env.close()


def blind(args, actor, reg, split):
    rows = []
    for spec in reg["splits"][split]:
        path = args.out / split / spec["id"] / "natural.json"
        row = read(path) if path.exists() else live(args, actor, spec, 50, path)
        rows.append(row)
        print(split, spec["id"], row["metrics"]["success"], row["metrics"]["steps"], flush=True)
        if split == "source" and len(rows) == 6 and not any(r["metrics"]["success"] for r in rows):
            raise RuntimeError("no source baseline successes in first six games")
    return rows


def draft_memory(row, cp, *, contract=None):
    payload = {"goal": cp["context"]["goal"], "stage":cp["stage"], "observation":cp["context"]["observation"],
               "original":cp["draft"]["action"], "correction":cp["preferred"][0]}
    def parse(_, x):
        if not isinstance(x.get("reminder"),str) or not 1 <= len(x["reminder"].split()) <= 70:
            raise ValueError("expected a short reminder")
        return x
    for style in ("chat","responses"):
        try:
            return teacher_request(payload, os.environ["OPENAI_BASE_URL"], "gpt-6-sol",
                                   style, instruction=TEACHER + (" Scope the wording to: " + contract
                                   if contract else ""), validate=parse)
        except TeacherCallError:
            if style == "responses":
                raise


def natural(args, spec):
    return read(args.out / spec["id"].split("-")[0] / spec["id"] / "natural.json")


def frozen_cap(args, reg):
    dev = [natural(args, s) for s in reg["splits"]["development"]]
    metrics = {str(c): sum(r["metrics"]["success"] and r["metrics"]["steps"] <= c
                           for r in dev) for c in reg["calls"]}
    return next((c for c in reg["calls"] if metrics[str(c)] >= len(dev)/2), None), metrics


def source_opportunities(args, reg, cap):
    for spec in reg["splits"]["source"]:
        row = natural(args, spec)
        steps = row["trace"]["steps"]
        if row["metrics"]["success"] and len(steps) <= cap:
            continue
        stages = set()
        for ix, cp in enumerate(steps[:cap-1]):
            if cp["stage"] not in ("take", "treat", "deliver") or cp["stage"] in stages:
                continue
            if cp["draft"]["action"] in cp["preferred"]:
                continue
            stages.add(cp["stage"])
            yield spec, row, {"checkpoint": cp, "prefix": steps[:ix]}


def source_bank(args, actor, reg):
    cap, dev = frozen_cap(args, reg)
    if cap is None:
        raise RuntimeError(f"development floor performance {dev}; no target outcomes accessed")
    bank, direct_count, direct_misses = [], 0, 0
    source_cap = 50
    for spec, row, anchor in source_opportunities(args, reg, source_cap):
        cp = anchor["checkpoint"]
        dest = args.out / "source" / spec["id"] / f"step-{cp['step']:02d}"
        replacement = cp["preferred"][0]
        save(dest/"anchor.json", anchor)
        direct = live(args, actor, spec, source_cap, dest/"direct-screen.json", anchor=anchor, replacement=replacement)
        direct_count += 1
        print("direct screen", spec["id"], cp["step"], direct["metrics"]["success"], flush=True)
        if not direct["metrics"]["success"]:
            direct_misses += 1
            if direct_misses >= 3 and not bank:
                break
            continue
        teacher = read(dest/"teacher.json") if (dest/"teacher.json").exists() else draft_memory(row, cp)
        save(dest/"teacher.json", teacher)
        text = teacher["candidate"]["reminder"]
        probe = live(args, actor, spec, source_cap, dest/"text-screen.json", anchor=anchor, reminder=text)
        if not probe["metrics"]["success"]:
            print("text screen failed", spec["id"], cp["step"], flush=True)
            continue
        reps = []
        for seed in (913, 457, 808):
            metrics = {}
            order = ["silent", "direct", "remind"]
            random.Random(seed).shuffle(order)
            for arm in order:
                result = live(args, actor, spec, source_cap, dest/arm/f"{seed}.json", anchor=anchor, seed=seed,
                              reminder=text if arm=="remind" else None,
                              replacement=replacement if arm=="direct" else None)
                metrics[arm] = result["metrics"]
            reps.append({"seed":seed, "metrics":metrics})
        quality = {a:paired_contrast(reps,a) for a in ("direct","remind")}
        admit = all(q["supports_gain"] for q in quality.values()) and any(
            r["metrics"]["remind"]["success"] > r["metrics"]["silent"]["success"] for r in reps)
        evidence = {"admitted": admit, "replicates": reps, "q": quality, "teacher": digest(teacher)}
        save(dest/"evidence.json", evidence)
        print("source K", spec["id"], cp["stage"], admit, quality, flush=True)
        if admit:
            bank.append({"id":digest([spec,cp,text]), "stage":cp["stage"], "text":text,
                         "source":spec["id"], "evidence":str((dest/"evidence.json").relative_to(args.out)),
                         "evidence_sha256":sha_file(dest/"evidence.json"), "q":quality})
        if len(bank) >= 2 or direct_count >= 10:
            break
    report = {"cap":cap, "source_cap":source_cap, "development_wins_by_cap":dev, "bank":bank,
              "source_direct_screens":direct_count, "source_direct_misses":direct_misses}
    save(args.out/"freeze.json", report)
    print("FROZEN", len(bank), cap, direct_count, flush=True)


def verify_greedy(args, actor, reg, *, generalized=False):
    """Qualify source memories for the exact greedy policy used in the target pilot."""
    old = read(args.out/"freeze.json")
    if (args.out/"target-protocol.json").exists() and not generalized:
        raise ValueError("target cohort already opened")
    policy = {"actor": "same greedy Qwen scaffold as target", "source_cap": 50,
              "replicate_definition": "two fresh restores, opposite execution order; deterministic reproducibility only",
              "admission": "original fails, correction wins, wording wins; positive net utility in both restores",
              "stochastic_screen": digest(old), "source_outcomes_visible_when_amended": True}
    stem = "general" if generalized else "greedy"
    if generalized:
        policy["contract"] = ("The task requires cleaning the held target object; a matching clean command using a "
                              "sinkbasin is legal now. This condition applies to any sink content. "
                              "The advice should clearly recommend attempting the available cleaning action.")
    save(args.out/(stem+"-protocol.json"),policy)
    bank = []
    for spec,row,anchor in source_opportunities(args,reg,50):
        cp = anchor["checkpoint"]
        parent = args.out/"source"/spec["id"]/f"step-{cp['step']:02d}"
        if generalized:
            if not (parent/"direct-screen.json").exists() or not read(parent/"direct-screen.json")["metrics"]["success"]:
                continue
            proposed = parent/"teacher-general.json"
            teacher = (read(proposed) if proposed.exists() else draft_memory(row, cp, contract=policy["contract"]))
            save(proposed,teacher)
        else:
            if not (parent/"text-screen.json").exists() or not read(parent/"text-screen.json")["metrics"]["success"]:
                continue
            teacher = read(parent/"teacher.json")
        note = teacher["candidate"]["reminder"]
        dest = parent/("general" if generalized else "deterministic")
        reps = []
        if generalized:
            trial = live(args,actor,spec,50,dest/"screen.json",anchor=anchor,reminder=note)
            if not trial["metrics"]["success"]:
                print("general text screen failed",spec["id"],flush=True)
                continue
        for repeat, order in enumerate((("silent","direct","remind"),("remind","direct","silent")),1):
            metrics, decisions = {},{}
            for arm in order:
                replay = live(args,actor,spec,50,dest/str(repeat)/f"{arm}.json",anchor=anchor,
                              replacement=cp["preferred"][0] if arm=="direct" else None,
                              reminder=note if arm=="remind" else None)
                metrics[arm] = replay["metrics"]
                decisions[arm] = replay["trace"]["steps"][cp["step"]-1]["draft"]["action"]
            reps.append({"repeat":repeat,"metrics":metrics,"actions":decisions})
        admitted = all(r["metrics"]["silent"]["success"]==0 and
                       all(r["metrics"][a]["success"]==1 and
                           r["metrics"][a]["utility"]>r["metrics"]["silent"]["utility"]
                           for a in ("direct","remind")) for r in reps)
        evidence = dest/"evidence.json"
        save(evidence,{"admitted":admitted,"replays":reps,"policy":policy,"teacher":digest(teacher)})
        print("greedy verification",spec["id"],admitted,[r["actions"] for r in reps],flush=True)
        if admitted:
            contract = {"stage":cp["stage"], "fixture":"sinkbasin",
                        "requires_empty_fixture_feedback": not generalized and "appears empty" in note.lower()}
            bank.append({"id":digest([spec, cp, note, contract]), "stage":cp["stage"], "text":note,
                         "source":spec["id"], "contract":contract, "evidence_sha256":sha_file(evidence),
                         "evidence":str(evidence.relative_to(args.out))})
    save(args.out/f"freeze-{stem}.json",{**old,"bank":bank,"admission":"deterministic", "protocol":policy})
    print("FROZEN_"+stem.upper(),len(bank),flush=True)


def eligible_card(checkpoint, prefix, card):
    if checkpoint["stage"] != card["stage"] or not checkpoint["preferred"]:
        return False
    contract = card.get("contract")
    if contract is None:
        return True
    fixture = contract["fixture"]
    if not any(f" with {fixture} " in a for a in checkpoint["preferred"]):
        return False
    if not contract["requires_empty_fixture_feedback"]:
        return True
    observations = [checkpoint["context"]["observation"], *[r["execution"]["observation"] for r in prefix]]
    return any(fixture in v.lower() and "see nothing" in v.lower() for v in observations)


def replay_budget(args, result, limit):
    """Grade exactly the prefix allowed by the shared actor-call budget."""
    extra = result.get("extra_draft")
    trace = result["trace"]
    rows = trace["steps"][:limit-int(extra is not None)]
    terminal = trace["end_state"] if len(rows) == len(trace["steps"]) else rows[-1]["execution"]["state"]
    ep = Episode(args.data, args.config, result["spec"])
    try:
        replay = ep.restore([r["execution"] for r in rows], terminal)
        try:
            prefix = {"success": int(replay.won), "steps": rows}
            return costs(prefix) if extra is None else costs(prefix, 0, extra)
        finally:
            replay.env.close()
    finally:
        ep.env.close()


def evaluate(args, actor, reg):
    bank_root = Path(reg["source"]) if "source" in reg else args.out
    greedy = bank_root/reg.get("source_bank","freeze-greedy.json")
    frozen = read(greedy if greedy.exists() else bank_root/"freeze.json")
    if reg.get("source_digest",digest(frozen)) != digest(frozen):
        raise ValueError("replication source bank changed")
    cap = frozen["cap"]
    if not frozen["bank"]:
        raise RuntimeError("no verified memory; stop before test")
    protocol = {"primary": cap, "shared_call_budgets": sorted(set(reg["calls"]+[50])),
                "policy": "first applicable stage before the cap, includes aligned drafts",
                "resources": "one additional actor call and its measured tokens/time for revision",
                "proof": "for smaller caps replay exact respective full-arm prefixes and grade with official won"}
    if "local_budgets" in reg:
        protocol.update(local_budgets=reg["local_budgets"], primary_local_budget=reg["primary_local_budget"])
    dest = args.out/"target-protocol.json"
    if dest.exists() and read(dest) != protocol:
        raise ValueError("target protocol changed")
    save(dest,protocol)
    for m in frozen["bank"]:
        if sha_file(bank_root/m["evidence"]) != m["evidence_sha256"]:
            raise ValueError("source evidence changed")
    pairs, no_match = [], []
    for spec in reg["splits"]["test"]:
        dest = args.out/"test"/spec["id"]
        npath = dest/"natural.json"
        n = read(npath) if npath.exists() else live(args, actor, spec, 50, npath)
        steps = n["trace"]["steps"]
        eligible = [(i,c,m) for i,c in enumerate(steps[:49])
                    for m in frozen["bank"] if eligible_card(c,steps[:i],m)]
        if not eligible:
            no_match.append({"id":spec["id"],"natural":n["metrics"]})
            continue
        i, cp, card = eligible[0]
        anchor = {"checkpoint":cp,"prefix":steps[:i]}
        save(dest/"anchor.json",{**anchor,"card":card["id"],"aligned":cp["draft"]["action"] in cp["preferred"]})
        arms = {}
        order = ["silent","remind"]
        random.Random(digest([spec["seed"],50])).shuffle(order)
        for arm in order:
            arms[arm] = live(args, actor, spec, 50, dest/"50"/f"{arm}.json", anchor=anchor,
                             reminder=card["text"] if arm=="remind" else None)
        if (arms["silent"]["metrics"]["success"] != n["metrics"]["success"] or
                arms["silent"]["trace"]["end_state"] != n["trace"]["end_state"] or
                [r["draft"]["action"] for r in arms["silent"]["trace"]["steps"]]
                != [r["draft"]["action"] for r in steps]):
            raise ValueError(f"silent replay does not reproduce blind natural rollout: {spec['id']}")
        budgets = [(b, "episode", b) for b in protocol["shared_call_budgets"]]
        budgets.extend((min(i+b,50), "checkpoint", b) for b in reg.get("local_budgets", []))
        for limit, scope, budget in budgets:
            if i + 2 > limit:
                continue
            silent, remind = (arms[a]["metrics"] if limit == 50 else replay_budget(args, arms[a], limit)
                              for a in ("silent","remind"))
            row = {"id":spec["id"],"step":cp["step"],"cap":limit,"aligned":cp["draft"]["action"] in cp["preferred"],
                   "silent":silent,"remind":remind,"delta":{k:remind[k]-silent[k] for k in silent},
                   "same_revision":arms["remind"]["trace"]["steps"][i]["draft"]["action"]==cp["draft"]["action"]}
            if "local_budgets" in reg:
                row.update(budget_scope=scope,budget=budget)
            pairs.append(row)
            print("TARGET", row["id"], limit, silent["success"], remind["success"], row["delta"]["tokens"], flush=True)
        save(args.out/"summary.json",{"frozen":frozen,"protocol":protocol,"pairs":pairs,"unmatched":no_match,
                                     "total":len(reg["splits"]["test"])})
    save(args.out/"summary.json",{"frozen":frozen,"protocol":protocol,"pairs":pairs,"unmatched":no_match,
                                 "total":len(reg["splits"]["test"]),"complete":True})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=["register", "register-replication", "source", "development", "verify",
                                       "verify-greedy", "verify-general", "test"])
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--weights", type=Path, required=True)
    cli.add_argument("--source-run", type=Path)
    cli.add_argument("--general", action="store_true")
    cli.add_argument("--seed", type=int, default=262711)
    cli.add_argument("--calls", type=int, nargs="+", default=[8,12,16,24,50])
    args = cli.parse_args()
    path = args.out / "manifest.json"
    if args.stage in {"register","register-replication"}:
        if path.exists():
            raise FileExistsError(path)
        if args.stage == "register-replication":
            register_replication(args)
        else:
            register(args)
        return
    reg = read(path)
    if sha_file(args.config) != reg["config"]:
        raise ValueError("changed environment configuration")
    actor = Scaffold(Actor(args.weights))
    if args.stage in {"source","development"}:
        blind(args, actor, reg, args.stage)
    elif args.stage == "verify":
        source_bank(args, actor, reg)
    elif args.stage == "verify-greedy":
        verify_greedy(args, actor, reg)
    elif args.stage == "verify-general":
        verify_greedy(args, actor, reg, generalized=True)
    else:
        evaluate(args, actor, reg)


if __name__ == "__main__":
    main()
