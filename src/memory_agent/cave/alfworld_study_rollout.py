"""Run the full-study actor on frozen data splits with only public inputs."""

import argparse
import contextlib
import io
from pathlib import Path

from .alfworld_env import Actor, Episode, MAX_STEPS, sha_file
from .alfworld_experiment import costs
from .alfworld_ranker import RankedActor, examples, load
from .records import digest, read, save


def receipt(root, spec, name="natural"):
    return root/spec["id"].rsplit("-",1)[0]/spec["id"]/(name+".json")


def rollout(args, spec, actor, *, anchor=None, reminder=None, correction=None, cap=MAX_STEPS,
            continuation_reminder=None):
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        ep = Episode(args.data,args.config,spec)
        if anchor:
            initial = ep
            try:
                ep = ep.restore([r["execution"] for r in anchor["prefix"]],anchor["state"])
            finally:
                initial.env.close()
    rows = list(anchor["prefix"]) if anchor else []
    initial_step = len(rows)
    add = None
    try:
        while not ep.terminal and len(rows)+int(add is not None)<cap:
            if anchor and len(rows)==initial_step:
                if (reminder is not None) and len(rows)+1>=cap:
                    raise ValueError("insufficient calls for redraft")
                draft = anchor["draft"]
                if reminder is not None:
                    add = draft
                    draft = actor.propose(ep,reminder,draft["action"])
                elif correction is not None:
                    draft = {**draft,"action":correction}
            else:
                draft = actor.propose(ep, continuation_reminder) if continuation_reminder else actor.propose(ep)
            row = {"observation":ep.observation,"goal":ep.goal,
                   "state":ep.signature(),"draft":draft,"public_menu":ep.commands}
            row["execution"] = ep.advance(draft["action"])
            rows.append(row)
        trace = {"success":int(ep.won),"steps":rows,"end_state":ep.signature()}
        bill = costs(trace) if add is None else costs(trace,initial_step,add)
        return {"spec":spec,"trace":trace,"metrics":bill,"extra_draft":add}
    finally:
        ep.env.close()


def run_natural(args, manifest, actor, group):
    queue = list(manifest["splits"][group])
    if group == "development":
        # Test one game of each task family early before committing GPU to the full development cohort.
        by_family = {}
        for s in queue:
            by_family.setdefault(s["family"],[]).append(s)
        queue = [family[i] for i in range(max(map(len,by_family.values()))) for family in by_family.values()
                 if i<len(family)]
    if args.limit is not None:
        queue = queue[:args.limit]
    good = 0
    for i,spec in enumerate(queue,1):
        path = receipt(args.out,spec)
        report = read(path) if path.exists() else rollout(args,spec,actor)
        if not path.exists():
            save(path,report)
        good += report["metrics"]["success"]
        print(group,spec["id"],report["metrics"]["success"],report["metrics"]["steps"],f"{good}/{i}",flush=True)
        if group=="development" and i==8 and not good:
            raise RuntimeError("no development baseline successes in first eight games")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=["development","historical","memories","memories-grounded",
                                      "memories-explore","memories-abstract","memories-diverse","pair-dev","test"])
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--data",type=Path,required=True)
    cli.add_argument("--config",type=Path,required=True)
    cli.add_argument("--weights",type=Path,required=True)
    cli.add_argument("--top",type=int,default=3)
    cli.add_argument("--repeat-aware",action="store_true")
    cli.add_argument("--examples",action="store_true")
    cli.add_argument("--indexed",action="store_true")
    cli.add_argument("--limit",type=int)
    cli.add_argument("--baseline-from",type=Path,help="Reuse the measured silent arm from an audited run")
    cli.add_argument("--source-relative-budget",action="store_true",
                     help="Also test a pair budget based on the matched card's verified source continuation")
    args = cli.parse_args()
    if args.baseline_from is not None and args.stage!="test":
        cli.error("--baseline-from requires the test stage")
    if args.source_relative_budget and args.stage not in ("pair-dev","test"):
        cli.error("--source-relative-budget requires pair-dev or test")
    manifest = read(args.out/"manifest.json")
    if sha_file(args.config)!=manifest["config_sha256"]:
        raise ValueError("environment changed")
    record = {"top":args.top,"actor":str(args.weights.resolve()),
              "ranker":sha_file(args.out/"ranker.pt")}
    if args.repeat_aware:
        record["repeat_aware"] = True
    if args.examples:
        record["examples"] = True
    if args.indexed:
        record["indexed"] = True
    lock = args.out/"actor.json"
    if lock.exists() and read(lock)!=record:
        raise ValueError("actor configuration changed")
    save(lock,record)
    ranker = load(args.out)
    actor = RankedActor(Actor(args.weights),ranker,args.top,args.repeat_aware,
                        examples(args.out,ranker) if args.examples else (),args.indexed)
    if args.stage in ("development","historical"):
        return run_natural(args,manifest,actor,args.stage)
    from .alfworld_study_pairs import create_memories, historical_exploration, pilot, full

    if args.stage=="memories-grounded":
        return create_memories(args,manifest,actor,grounded=True)
    if args.stage=="memories-explore":
        return historical_exploration(args,manifest,actor)
    if args.stage in ("memories-abstract","memories-diverse"):
        return historical_exploration(args,manifest,actor,abstract=True,diverse=args.stage=="memories-diverse")
    return {"memories":create_memories,"pair-dev":pilot,"test":full}[args.stage](args,manifest,actor)


if __name__=="__main__":
    main()
