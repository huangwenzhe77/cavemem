"""Verify sealed ALFWorld pairs and summarize one task once at each budget."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .alfworld_env import sha_file
from .records import read, save


def stats(rows):
    if not rows:
        return {"n": 0}
    kinds = [(r["silent"]["success"], r["remind"]["success"]) for r in rows]
    deltas = [r["delta"] for r in rows]
    joint = [r["id"] for r in rows if r["delta"]["success"] < 0
             and r["delta"]["tokens"] > 0 and r["delta"]["seconds"] > 0]
    both_more = [r["id"] for r in rows if r["silent"]["success"] == r["remind"]["success"] == 1
                 and r["delta"]["tokens"] > 0 and r["delta"]["seconds"] > 0]
    return {"n":len(rows), "silent_success":sum(y[0] for y in kinds), "remind_success":sum(y[1] for y in kinds),
            "harmed":kinds.count((1,0)), "rescued":kinds.count((0,1)), "both_success":kinds.count((1,1)),
            "both_fail":kinds.count((0,0)), "worse_and_more_tokens_and_time":joint,
            "both_success_more_tokens_and_time":both_more,
            "mean_delta":{k:mean(d[k] for d in deltas) for k in deltas[0]}}


def audit(root, source):
    reg, summary = read(root/"manifest.json"), read(root/"summary.json")
    assert summary["complete"]
    for card in summary["frozen"]["bank"]:
        path = source/card["evidence"]
        assert sha_file(path)==card["evidence_sha256"]
        evidence = read(path)
        assert evidence["admitted"]
        for r in evidence["replays"]:
            assert [r["metrics"][x]["success"] for x in ("silent","direct","remind")] == [0,1,1]
    population = {x["id"] for x in reg["splits"]["test"]}
    assert len(population)==summary["total"]
    naturals = {i:read(root/"test"/i/"natural.json") for i in population}
    assert set(naturals)==population
    groups = defaultdict(list)
    for r in summary["pairs"]:
        assert r["id"] in population
        assert r["delta"]=={k:r["remind"][k]-r["silent"][k] for k in r["silent"]}
        scope, budget = r.get("budget_scope","episode"), r.get("budget",r["cap"])
        assert r["silent"]["actor_calls"] <= r["cap"] and r["remind"]["actor_calls"] <= r["cap"]
        if r["cap"]==50:
            trials = {a:read(root/"test"/r["id"]/"50"/f"{a}.json") for a in ("silent","remind")}
            assert trials["silent"]["trace"]["end_state"]==naturals[r["id"]]["trace"]["end_state"]
            for a,t in trials.items():
                assert t["metrics"]==r[a]
            i=r["step"]-1
            assert r["same_revision"] == (trials["silent"]["trace"]["steps"][i]["draft"]["action"]==
                                           trials["remind"]["trace"]["steps"][i]["draft"]["action"])
        groups[scope,budget].append(r)
    output = {"n_tasks":len(population), "natural_wins":sum(n["metrics"]["success"] for n in naturals.values()),
              "matched_tasks":len({r["id"] for r in summary["pairs"]}),"groups":{}}
    for (s,b), rows in sorted(groups.items()):
        ids = {r["id"] for r in rows}
        assert len(ids)==len(rows)
        group = {"all_matched":stats(rows),"original_aligned":stats([r for r in rows if r["aligned"]]),
                 "original_not_aligned":stats([r for r in rows if not r["aligned"]])}
        if s=="episode":
            baseline = sum(n["metrics"]["success"] and n["metrics"]["steps"] <= b
                           for i,n in naturals.items() if i not in ids)
            group["full_cohort_success"] = {a:baseline+sum(r[a]["success"] for r in rows)
                                            for a in ("silent","remind")}
        output["groups"][f"{s}:{b}"]=group
    save(root/"audited.json",output)
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--source",type=Path)
    args=parser.parse_args()
    print(audit(args.root,args.source or args.root))
