"""A preregistered single-factor study of residual actor-call budget on natural mini tasks."""
import argparse
import math
import random
from collections import Counter
from pathlib import Path

from . import formal_plan as fp, gate_validation as gv, prospective, runtime_pair
from .branches import load_actor
from .budget_fork import fork
from .constrained_gate import Policy
from .experiment import archive_digest
from .gate import lines, signed
from .models import xml_parser
from .records import digest, encode, read, save, verify
from .replay import replay_tape, require
from .supplement_plan import SHAPES, make_input


def collect_inputs(obj, seen):
    if isinstance(obj,dict):
        values=obj.get("input")
        if isinstance(values,list) and values and all(type(v) is int for v in values):
            seen.add(tuple(sorted(values)))
        for v in obj.values():
            collect_inputs(v,seen)
    elif isinstance(obj,list):
        for row in obj:
            collect_inputs(row,seen)


def register(root):
    parent=root.parent/"cave-runtime-pair-01"
    previous=signed(parent/"preregistration.json","plan_id")
    seen=set()
    for name in ("preregistration.json","manifest.json","sealed-test.json"):
        for path in root.parent.glob("cave-*/**/"+name):
            collect_inputs(read(path),seen)
    # Live transfer smoke included a small number of new rows in pre-state receipts only.
    for path in (root.parent/"cave-transfer-01").glob("**/001/pre.json"):
        collect_inputs(read(path),seen)
    original=digest(sorted(seen))
    rows=[]
    for i,(name,(family,text)) in enumerate(gv.TEMPLATES.items()):
        seed=8100000+i*10000
        for n in range(8):
            shape=SHAPES[n%4]
            while True:
                seed+=1
                values=make_input(seed,shape,family)
                if values is not None and not any(fp.near_duplicate(values,t) for t in seen):
                    break
            seen.add(tuple(sorted(values)))
            task=dict(template_id=name,family=family,instruction=text,input=values)
            rows.append(dict(episode_id=f"ep-{len(rows):04d}",seed=seed,task=task,task_digest=digest(task),input_stratum=shape))
    p={k:v for k,v in previous.items() if k not in ("plan_id","parent_plan_id","schema","prior_inputs_digest","design","created_utc")}
    p.update(schema="budget-dev/v1",parent_plan_id=previous["plan_id"],prior_inputs_digest=original,
             cohorts=[{"split":"development","tasks":rows}], code=fp.code_digest("budget_dev.py"),
             fork_code=fp.code_digest("budget_fork.py"), seed=8100000,
             design="32 new tasks. First-hit Bernoulli(.5) on reached applicable affordable drafts with >=2 normal remaining calls; no refill. A/A plus 4 actual arms at each selected prefix.",
             budgets={"normal":"4 - number of actor calls through a0", "one":1},
             outcomes=["Y", "D", "H", "rescue", "both_success_extra_tokens", "both_success_extra_seconds",
                       "H_and_extra_tokens_and_extra_seconds","D_one_minus_normal"],
             time_measure="common prefix generation plus new generation and tool execution; omit shared selection and A/A")
    root.mkdir(parents=True,exist_ok=False)
    save(root/"preregistration.json",{**p,"plan_id":digest(p)})
    print("registered",len(rows),digest(p),flush=True)


def load(root):
    p=signed(root/"preregistration.json","plan_id")
    require(p["code"]==fp.code_digest("budget_dev.py") and p["fork_code"]==fp.code_digest("budget_fork.py")
            and p["runtime_code"]==fp.code_digest("runtime_pair.py"),"registration changed")
    return p


def blind(root):
    p=load(root)
    run=root/"blind"
    run.mkdir(exist_ok=False)
    tasks=p["cohorts"][0]["tasks"]
    mf=dict(run_id=digest([p["plan_id"],"blind"]),split="development",schema=1,benchmark="mini-data-v1",
            actor=p["actor"],tasks=[r["task"] for r in tasks],max_steps=4,cohort="natural",
            memory_enabled=False,teacher_enabled=False)
    save(run/"manifest.json",mf)
    actor=load_actor(mf)
    cards=prospective.verified_cards(p,fp.DEFAULT_BANK,fp.DEFAULT_PACKAGE,root.parent/"cave-pilot-01/preregistration.json")
    policy=Policy(root.parent/"cave-primal-dual-01/model.json",root.parent/"cave-mlp-01/frozen/model.json")
    parser=xml_parser()
    # runtime_pair.episode consults this manifest solely for revision_fits and run identity.
    # Its factual rollout horizon comes from p (4). The eligibility check needs two spare calls.
    fit_limits={**mf,"max_steps":3}
    with (run/"index.jsonl").open("x",encoding="utf8") as f:
        for t in tasks:
            row=runtime_pair.episode(run,t,fit_limits,p,actor,parser,cards,policy)
            f.write(encode(row)+"\n")
            f.flush()
            print("blind",row["episode_id"],row["selected"],flush=True)
    rows=lines(run/"index.jsonl")
    save(root/"seal.json",{"plan_id":p["plan_id"],"manifest":digest(mf),"archives":[archive_digest(run,r) for r in rows]})
    return pair(root,actor)


def pair(root,actor=None):
    p=load(root)
    run=root/"blind"
    mf=read(run/"manifest.json")
    rows=lines(run/"index.jsonl")
    seal=read(root/"seal.json")
    require(seal=={"plan_id":p["plan_id"],"manifest":digest(mf),"archives":[archive_digest(run,r) for r in rows]},"unsealed input")
    actor=actor or load_actor(mf)
    for row in rows:
        cp=row["selected"]
        if cp is None:
            continue
        out=root/"pairs"/row["episode_id"]
        d=read(run/cp/"prediction.json")
        require(d["turn"] in (1,2) and d["rho"]==1 and d["v"]==1,"no budget contrast")
        if not (out/"aa.json").exists():
            replay_tape(run,mf,row,start=d["turn"])
            aa=prospective.check_free(out/"aa",run,mf,row,d,p,actor)
            save(out/"aa.json",aa)
        if read(out/"aa.json")["free_status"]!="agree":
            continue
        seed=random.Random(f"coupled-budget:{row['seed']}").randrange(2**31)
        order=[(level,arm) for level in ("normal","one") for arm in ("silent","remind")]
        random.Random(f"order:{row['seed']}").shuffle(order)
        for level,arm in order:
            path=out/level/(arm+".json")
            if path.exists():
                continue
            remaining=4-d["turn"] if level=="normal" else 1
            raw=fork(run,mf,row,d["turn"],actor,arm,d["reminder"],seed,p["total_tokens"],p["total_tools"],remaining)
            save(path,raw)
        print("paired",cp,flush=True)
    audit(root)


def audit(root):
    p=load(root)
    run=root/"blind"
    rows=lines(run/"index.jsonl")
    manifest=read(run/"manifest.json")
    require(len(rows)==32 and read(root/"seal.json")=={"plan_id":p["plan_id"],"manifest":digest(manifest),
            "archives":[archive_digest(run,r) for r in rows]},"invalid sealed denominator")
    counts={t:Counter() for t in ("normal","one")}
    comparisons=[]
    for row in rows:
        cp=row["selected"]
        if cp is None:
            continue
        d=read(run/cp/"prediction.json")
        folder=root/"pairs"/row["episode_id"]
        if read(folder/"aa.json")["free_status"]!="agree":
            continue
        obs={}
        common=None
        for level in counts:
            obs[level]={}
            for name in ("silent","remind"):
                a=read(folder/level/(name+".json"))
                u=a["usage"]
                remain=a["limits"]["actor_calls"]-a["common_prefix"]["actor_calls"]
                require(a["limits"]["turns"]==4 and a["limits"]["total_tokens"]==p["total_tokens"]
                        and a["limits"]["tool_calls"]==p["total_tools"] and remain==(4-d["turn"] if level=="normal" else 1)
                        and a["a0_digest"]==d["a0_digest"] and u["actor_calls"]<=a["limits"]["actor_calls"]
                        and a["revision_calls"]==int(name=="remind") and a["arm"]==name,"invalid intervention")
                identity=(a["seed"],a["common_prefix"])
                if common is None:
                    common=identity
                require(common==identity,"unequal prefix or seed")
                y=a["score"]["reward"]
                if y is not None:
                    require(y==verify(row["task"],a["state"])["reward"],"incorrect independent grade")
                total=u["input_tokens"]+u["output_tokens"]
                sec=u["common_actor_seconds"]+u["new_actor_seconds"]+u["new_tool_seconds"]
                penalty=p["lambda_tokens"]*total/1000+p["lambda_tool"]*u["tool_calls"]+p["lambda_time"]*sec/60
                obs[level][name]=dict(y=y,tokens=total,seconds=sec,utility=y-penalty if y is not None else None,
                                      initial_remaining_calls=remain,actual_new_calls=u["actor_calls"]-a["common_prefix"]["actor_calls"],
                                      termination=a["termination"])
            s,r=obs[level]["silent"],obs[level]["remind"]
            c=counts[level]
            c["assigned_pairs"]+=1
            if s["y"] is None or r["y"] is None:
                c["missing"]+=1
                continue
            cell={(1,1):"both_success",(1,0):"regression",(0,1):"rescue",(0,0):"both_fail"}[s["y"],r["y"]]
            c[cell]+=1
            dt=r["tokens"]-s["tokens"]
            ds=r["seconds"]-s["seconds"]
            du=r["utility"]-s["utility"]
            c["extra_tokens"]+=dt
            c["extra_seconds"]+=ds
            c["utility_gain"]+=du
            for suffix,indicator in (("extra_tokens",dt>0),("extra_seconds",ds>0),("both_costs_higher",dt>0 and ds>0)):
                c[cell+"_"+suffix]+=int(indicator)
            obs[level].update(cell=cell,delta_tokens=dt,delta_seconds=ds,D=du)
        comparisons.append({"episode_id":row["episode_id"],"checkpoint":cp,"a0_digest":d["a0_digest"],"memory_id":d["memory_id"],
                            "by_budget":obs,"D_one_minus_normal":obs["one"]["D"]-obs["normal"]["D"] if all("D" in v for v in obs.values()) else None})
    save(root/"paired-metrics.json",comparisons)
    contrasts=[r["D_one_minus_normal"] for r in comparisons if r["D_one_minus_normal"] is not None]
    report=dict(plan_id=p["plan_id"],tasks=len(rows),risk_tasks=sum(r["eligible_draws"]>0 for r in rows),
                selected=sum(r["selected"] is not None for r in rows),pairs_per_budget=counts,
                n_contrasts=len(contrasts),mean_change_in_D=sum(contrasts)/len(contrasts) if contrasts else None,
                sample_digest=digest(comparisons))
    save(root/"analysis.json",report)
    print(report,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("command",choices=["register","blind","pair","audit"])
    cli.add_argument("--root",type=Path,default=Path("outputs/cave-budget-dev-01"))
    args=cli.parse_args()
    {"register":register,"blind":blind,"pair":pair,"audit":audit}[args.command](args.root)

