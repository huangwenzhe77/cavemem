"""Audit coverage and quantify paired uncertainty on the registered game instances."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .alfworld_env import sha_file
from .alfworld_study_pairs import aggregate
from .alfworld_study_rollout import receipt
from .records import digest, read, save


def effects(rows, templates, seed, *, aligned_only=False):
    by_setting = defaultdict(list)
    for row in rows:
        for score in row["scores"]:
            trigger = (not aligned_only or bool(score["eligible"] and row["match"]["aligned"]))
            by_setting[score["setting"]].append((row["spec"],score,trigger))
    out = {}
    for setting, pairs in by_setting.items():
        values = defaultdict(list)
        for task,p,trigger in pairs:
            s,m=p["silent"],p["remind"]
            if not trigger:
                m = s
            values[templates[task]].append([m[k]-s[k] for k in ("success","tokens","seconds","actor_seconds")])
        per_template = list(values.values())
        lengths = np.array([len(x) for x in per_template])
        totals = np.array([np.sum(x,axis=0) for x in per_template])
        gen = np.random.default_rng(seed)
        draws = gen.integers(len(totals),size=(3000,len(totals)))
        sample = totals[draws].sum(axis=1)/lengths[draws].sum(axis=1)[:,None]
        applied = totals.sum(axis=0)/lengths.sum()
        out[setting] = {"templates":len(totals),"mean":applied.tolist(),
                        "cluster_bootstrap_95":np.quantile(sample,[.025,.975],axis=0).T.tolist()}
    return out


def audit(root):
    manifest = read(root/"manifest.json")
    bank = read(root/"bank.json")
    frozen = read(root/"test-freeze.json")
    summary = read(root/"summary.json")
    assert frozen["bank"]==digest(bank) and frozen==summary["freeze"]
    assert bank["cards"]
    for m in bank["cards"]:
        evidence = root/m["evidence"]
        assert sha_file(evidence)==m["sha256"]
        arms = read(evidence)["metrics"]
        assert len(arms)==2
        assert all([r[z]["success"] for z in ("silent","direct","remind")]==[0,1,1] for r in arms)
    groups = {}
    templates = {}
    settings = {"50",str(frozen["cap"]),f"checkpoint:{frozen['local_budget']}"}
    if frozen.get("source_relative_budget",False):
        settings.add("source-relative")
    for split, n in frozen["split_sizes"].items():
        specs = manifest["splits"][split]
        assert len(specs)==n
        group = groups[split] = []
        for s in specs:
            natural = read(receipt(root,s))
            row = read(receipt(root,s,"paired"))
            assert natural["spec"]==s and row["spec"]==s["id"]
            templates[s["id"]] = s["template"]
            groups_seen = set()
            for sc in row["scores"]:
                assert sc["setting"] not in groups_seen
                groups_seen.add(sc["setting"])
                for arm in ("silent","remind"):
                    assert sc[arm]["actor_calls"]<=sc["cap"]
                    assert sc[arm]["success"] in (0,1)
                if not sc["eligible"]:
                    assert sc["silent"]==sc["remind"]
                if sc["setting"]=="50":
                    assert sc["silent"]==natural["metrics"]
            assert settings==groups_seen
            group.append(row)
        assert aggregate(group)==summary["groups"][split]
    overall = [r for v in groups.values() for r in v]
    assert len(overall)==274 and len(templates)==274
    assert aggregate(overall)==summary["overall"]
    result = {"n":len(overall),"verified_memory_cards":len(bank["cards"]),
              "metric_order":["win_probability","tokens","total_seconds","actor_seconds"],
              "groups":{n:effects(rows,templates,manifest["seed"]) for n,rows in groups.items()},
              "overall":effects(overall,templates,manifest["seed"]),
              "aligned_only_overall":effects(overall,templates,manifest["seed"],aligned_only=True),
              "summary":summary}
    save(root/"audited.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    audited=audit(args.root)
    print("AUDITED",audited["n"],audited["verified_memory_cards"])
