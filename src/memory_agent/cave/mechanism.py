"""Collect a prospectively defined, prefix-enriched supplementary training cohort."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from . import formal, formal_plan as fp, mechanism_selector as selector, prospective as pros
from .branches import load_actor
from .experiment import archive_digest
from .pilot_report import paired_obs
from .records import digest, read, save
from .replay import replay_tape, require, restore


def context(root):
    plan, pid = fp.load(root)
    study = read(root/"mechanism/preregistration.json")
    require(study["study_id"] == digest({k:v for k,v in study.items() if k!="study_id"})
            and study["plan_id"] == pid and study["selector_sha"] == fp.code_digest("mechanism_selector.py"),
            "secondary registration changed")
    manifest, rows = fp.checked_run(plan, "train", root/"train")
    return plan, pid, study, manifest, rows


def choose(root):
    plan, pid, study, manifest, rows = context(root)
    require(not (root/"audit-train").exists(), "choose before measuring any primary branches")
    cards = pros.verified_cards(plan, fp.DEFAULT_BANK, fp.DEFAULT_PACKAGE, fp.OUTPUT/"cave-pilot-01/preregistration.json")
    choices = selector.choose(root/"train", rows, plan, cards)
    path = root/"mechanism/selection.json"
    require(not path.exists(), "selection already frozen")
    seal = {"study_id": study["study_id"], "run_id": manifest["run_id"],
            "blind_digest": digest(read(root/"train/blind-seal.json")), "rows": choices}
    save(path,{**seal,"selection_id":digest(seal)})
    print("frozen",dict(Counter(d.get("mechanism","none") for d in choices)))


def selection(root, study, manifest):
    obj = read(root/"mechanism/selection.json")
    require(obj["selection_id"] == digest({k:v for k,v in obj.items() if k!="selection_id"})
            and obj["study_id"] == study["study_id"] and obj["run_id"] == manifest["run_id"]
            and obj["blind_digest"] == digest(read(root/"train/blind-seal.json")), "selection changed")
    return obj


def collect(root):
    plan, pid, study, mf, rows = context(root)
    selected = selection(root, study, mf)
    run = root/"train"
    primary = read(root/"audit-train/pairs/matches.json")
    read(root/"audit-train/pairs/pairs.json")
    dest = root/"mechanism/audit"
    dest.mkdir(exist_ok=False)
    subset, decisions, checks = [], [], []
    actor = None
    for row, d, old in zip(rows, selected["rows"], primary, strict=True):
        eid = row["episode_id"]
        require(eid == d["episode_id"] == old["episode_id"], "episode alignment")
        if "mechanism" not in d or d["checkpoint_id"] == old.get("checkpoint_id"):
            continue
        if actor is None:
            actor = load_actor(mf)
        a = {"episode_id":eid, "tape_anchor":d["turn"], "archive_digest":archive_digest(run,row)}
        try:
            tapes=[]
            for side in ("a","b"):
                tape = replay_tape(run,mf,row,start=d["turn"])
                save(dest/"aa"/eid/f"tape-{side}.json",tape)
                tapes.append(tape)
            require(tapes[0]["trace_digest"] == tapes[1]["trace_digest"] and
                    tapes[0]["score"]["reward"] == tapes[1]["score"]["reward"], "tape A/A disagreed")
            a["tape_status"]="pass"
            a.update(pros.check_free(dest/"aa"/eid,run,mf,row,d,plan,actor))
        except (ValueError,KeyError,OSError) as exc:
            a.setdefault("tape_status","unavailable")
            a.update(free_status="unavailable",reason=str(exc))
        subset.append(row)
        decisions.append(d)
        checks.append(a)
    save(dest/"aa/checks.json",checks)
    policy = {**read(root/"audit-train/pairs/protocol.json"),"design":"mechanism-prefix-only",
              "selection_id":selected["selection_id"],"audit_id":digest(checks)}
    save(dest/"pairs/protocol.json",policy)
    save(dest/"pairs/matches.json",decisions)
    pros.collect_pairs(dest/"pairs",run,mf,subset,decisions,checks,plan,policy,actor)


def export(root):
    plan, pid, study, mf, rows = context(root)
    selected = selection(root, study, mf)
    by_episode = {r["episode_id"]:r for r in rows}
    by_mechanism = {d["checkpoint_id"]:d["mechanism"] for d in selected["rows"] if "mechanism" in d}
    samples, targets, evidence = [], defaultdict(list), []
    cells, counts, seen = defaultdict(Counter), Counter(), set()
    for design, base in (("uniform",root/"audit-train"),("mechanism",root/"mechanism/audit")):
        pdir = base/"pairs"
        decisions = {d["episode_id"]:d for d in read(pdir/"matches.json")}
        aa = read(base/"aa/aa.json")["checks"] if design=="uniform" else read(base/"aa/checks.json")
        checks = {a["episode_id"]:a for a in aa}
        policy = read(pdir/"protocol.json")
        for raw in read(pdir/"pairs.json"):
            eid=raw["episode_id"]
            if "pair_id" not in raw:
                counts["excluded_"+raw["status"]]+=1
                continue
            d,row=decisions[eid],by_episode[eid]
            obs=paired_obs(raw,pdir,plan)
            labels=formal.check_observation(pdir,raw,obs,d,plan,policy,row["task"],checks[eid],root/"train",row)
            if labels is None:
                counts["incomplete"]+=1
                continue
            checkpoint=d["checkpoint_id"]
            require(checkpoint not in seen,"duplicated training checkpoint")
            seen.add(checkpoint)
            sid=digest([pid,"train",mf["run_id"],checkpoint,raw["pair_id"]])
            pre,draft,_=restore(root/"train",row,d["turn"])
            sample={"sample_id":sid,"features":formal.gate_inputs(row,pre,draft,d,plan),"targets":labels}
            samples.append(sample)
            delta=obs["delta"]; cell=obs["cell"]
            extra_cost=delta["success"]-labels["D"]
            costly=delta["total_tokens"]>0 and delta["accounted_seconds"]>0 and extra_cost>0
            target="costly-regression" if costly and cell=="regression" else "costly-both-pass" if costly and cell=="both_pass" else None
            if target:
                targets[target].append(sample)
            cells[design][cell]+=1
            if checkpoint in by_mechanism:
                cells["all_qualifying_including_overlap"][cell]+=1
                counts["overlap_complete"]+=int(design=="uniform")
            evidence.append({"sample_id":sid,"pair_id":raw["pair_id"],"primary_or_new":design,
                "mechanism":by_mechanism.get(checkpoint),"checkpoint":checkpoint,"cell":cell,"target":target,
                "time_delta":delta["accounted_seconds"],"token_delta":delta["total_tokens"],"cost_delta":extra_cost,
                "D":labels["D"],"H":labels["H"],"silent":obs["arms"]["silent"],"remind":obs["arms"]["remind"],
                "files":{a:(pdir/raw["arms"][a]["file"]).relative_to(root).as_posix() for a in ("silent","remind")}})
    dest=root/"deduplicated-training"
    dest.mkdir(exist_ok=False)
    formal.write_rows(dest/"all-pairs.jsonl",samples)
    requested = [sample for sample, entry in zip(samples, evidence, strict=True) if entry["target"]]
    formal.write_rows(dest/"target-examples.jsonl", requested)
    for name in ("costly-regression","costly-both-pass"):
        formal.write_rows(dest/(name+".jsonl"),targets[name])
    formal.write_rows(dest/"evidence.jsonl",evidence)
    summary={"plan_id":pid,"study_id":study["study_id"],"primary_tasks":len(rows),
             "secondary_selected":dict(Counter(by_mechanism.values())),"complete_unique":len(samples),
             "unique_tasks_with_pairs":len({e["checkpoint"].split("/")[0] for e in evidence}),
             "implementation_counts":dict(counts),"cells":dict(cells),
             "target_counts":{k:len(v) for k,v in targets.items()},"target_examples_digest":digest(requested),
             "all_pairs_digest":digest(samples),
             "target_digests":{k:digest(v) for k,v in targets.items()},"evidence_digest":digest(evidence)}
    save(dest/"summary.json",summary)
    print(summary["complete_unique"],summary["target_counts"])


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("mode",choices=("select","collect","export"))
    cli.add_argument("--root",required=True,type=Path)
    args=cli.parse_args()
    {"select":choose,"collect":collect,"export":export}[args.mode](args.root)
