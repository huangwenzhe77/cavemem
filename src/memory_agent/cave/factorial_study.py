"""Controlled applicability by exposure experiment and target-frequency standardization."""

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np

from . import formal_plan as fp, gate, gate_validation as gv, mini, single_arm as sa
from .branches import branch
from .comparison_cohort import priced
from .experiment import match
from .models import FrozenQwen, SYSTEM, xml_parser
from .records import digest, read, save, verify
from .replay import parse, require


def action(op):
    return '<response><analysis></analysis><plan></plan><commands><keystrokes duration="0">'+op+'</keystrokes></commands><task_complete>false</task_complete></response>'


def prefix(task, eid, rho, exposure, parser):
    start = mini.initial(task)
    prior = "ASC" if rho else "SHOW"
    op = "POSITIVE" if exposure else "SUBMIT"
    first,_ = parse(action(prior),parser)
    messages=[{"role":"system","content":SYSTEM},{"role":"user","content":task["instruction"]+
              "\nCurrent working list: "+str(start["working"])},{"role":"assistant","content":action(prior)},
              {"role":"user","content":"\n".join(mini.execute(start, first["commands"]))}]
    a,err=parse(action(op),parser)
    pre={"checkpoint_id":eid+"/001","state":start,"messages":messages,"used_input_tokens":0,
         "used_output_tokens":0,"used_tool_calls":1,"step":1,"remaining_steps":3}
    draft={"a0":a,"error":err,"a0_digest":digest(a),"warning":None,
           "used_input_tokens":0,"used_output_tokens":0,"used_tool_calls":1}
    return pre,draft,{"text":action(op),"seconds":0.}


def adversarial_bank(card):
    result={"verified":card}
    for name,text in (("unverified","Check that all input records are retained before submitting."),
                      ("contradictory","Remove zero and negative records before submitting the sorted list.")):
        body={"candidate":{**card["candidate"],"reminder":text},"source_memory":card["memory_id"],
              "evidence_status":"unverified","noise":name}
        result[name]={**body,"memory_id":digest(body)}
    return result


def vetted(card, verified_ids):
    return card["memory_id"] in verified_ids and digest({k:v for k,v in card.items() if k!="memory_id"})==card["memory_id"]


def run(root,outputs,n):
    require(n>=2,"two independent inputs required")
    plan, pid, _, cards=sa.validate(outputs/"cave-single-arm-01",fp.DEFAULT_BANK,fp.DEFAULT_PACKAGE)
    bank=adversarial_bank(cards["ascending"])
    require(vetted(bank["verified"],{cards["ascending"]["memory_id"]}),"invalid source")
    kinds=["silent",*bank,"neutral_recheck"]
    specs=[]
    for i in range(n):
        values=mini.task("controlled_transfer",8294100+i)["input"]
        values[:2]=[3,-1]
        require(min(values)<=0 and values!=sorted(values),"fixture must permit record loss")
        specs.append(dict(family="ascending",instruction=mini.TEMPLATES["controlled_transfer"][1],input=values))
    reg={"parent":pid,"specs":specs,"arms":kinds,"bank":bank,"code":fp.code_digest("factorial_study.py"),
         "cells":[[a,b] for a in (0,1) for b in (0,1)],
         "design":"Scripted prefix and cached action. Actor handles all real revisions and later steps. "
                  "Three model calls remain, one tool already consumed. Identical random stream within each cell."}
    root.mkdir(exist_ok=False)
    sa.durable(root/"preregistration.json",{**reg,"id":digest(reg)})
    actor=FrozenQwen(plan["actor"]["actor"],plan["actor"]["max_new_tokens"],0)
    require(actor.info==plan["actor"],"actor drift")
    parser=xml_parser()
    mf={"actor":actor.info,"max_steps":3,"cohort":"controlled_fixed_prefix"}
    rows=[]
    for i,task in enumerate(specs):
        for rho,e in reg["cells"]:
            eid=f"input-{i:03d}-{rho}{e}"
            path=root/"snapshots"/eid/"001"
            pre,draft,gen=prefix(task,eid,rho,e,parser)
            d=match([bank["verified"]],task,pre,draft)
            require((d["rho"],d["e"])==(rho,e),"constructed factors failed independent contract")
            for name,obj in (("pre",pre),("draft",draft),("generation",gen)):
                sa.durable(path/(name+".json"),obj)
            item={"episode_id":eid,"task":task,"steps":[{"turn_id":1,"checkpoint_id":eid+"/001"}]}
            obs={}
            order=list(kinds)
            random.Random("order:"+eid).shuffle(order)
            for kind in order:
                note=("Review your proposed action before proceeding." if kind=="neutral_recheck" else
                      bank[kind]["candidate"]["reminder"] if kind in bank else "")
                arm="silent" if kind=="silent" else "remind"
                raw=branch(root/"snapshots",mf,item,1,actor,arm,note,8294100+i,plan["total_tokens"],plan["total_tools"])
                require(raw["a0_digest"]==digest(draft["a0"]) and verify(task,raw["state"])["reward"]==raw["score"]["reward"],
                        "factorial outcome mismatch")
                sa.durable(root/"arms"/eid/(kind+".json"),raw)
                u=raw["usage"]
                obs[kind]=priced(plan,raw["score"]["reward"],u["input_tokens"]+u["output_tokens"],u["tool_calls"],
                                 u["new_actor_seconds"]+u["new_tool_seconds"])
            rows.append({"cluster":i,"cell":f"{rho}{e}","outcomes":obs,"order":order})
            print(eid,flush=True)
    sa.durable(root/"results.json",{"plan_id":digest(reg),"rows":rows,"digest":digest(rows)})


def standardize(cells,counts):
    denominator=sum(counts.values())
    covered=sum(n for k,n in counts.items() if k in cells)
    partial=(sum((counts[k]*np.array(v) for k,v in cells.items()),start=np.zeros(4))/covered
             if covered else None)
    return {"coverage":covered/denominator if denominator else 0.,
            "covered_only":partial.tolist() if covered else None,
            "population":partial.tolist() if covered and covered==denominator else None}


def report(root,outputs):
    reg=gate.signed(root/"preregistration.json","id")
    raw=read(root/"results.json")
    require(raw["plan_id"]==reg["id"] and raw["digest"]==digest(raw["rows"]),"factorial lineage")
    cells=["00","01","10","11"]
    reference=outputs/"cave-runtime-pair-01"/"blind"
    counts=Counter()
    for row in gate.lines(reference/"index.jsonl"):
        if row["task"]["family"]!="ascending": continue
        step=next((s for s in row["steps"] if s["executable"]),None)
        if step:
            h=read(reference/step["checkpoint_id"]/"hazard.json")
            label=f"{h['rho']}{h['e']}" if h.get("rho") in (0,1) and h.get("e") in (0,1) else "unknown"
        else: label="no_executable"
        counts[label]+=1
    summaries={}
    for kind in reg["arms"][1:]:
        groups={}
        verified_ids={reg["bank"]["verified"]["memory_id"]}
        for r in raw["rows"]:
            v=r["outcomes"][kind]
            s=r["outcomes"]["silent"]
            groups[(r["cluster"],r["cell"])]=[v["r"]-s["r"],int(v["y"]>s["y"]),int(v["y"]<s["y"]),v["tokens"]-s["tokens"]]
        n=len(reg["specs"])
        matrix=np.array([[groups[(i,c)] for c in cells] for i in range(n)])
        # Resample whole inputs, retaining all four cells and both outcomes within each input.
        indices=np.random.default_rng(8294).integers(0,n,(500,n))
        boot=matrix[indices].mean(axis=1)
        means=dict(zip(cells,matrix.mean(axis=0).tolist()))
        summaries[kind]={"forced_cell_means":means,
                         "cell_bootstrap95":dict(zip(cells,np.quantile(boot,[.025,.975],axis=0).transpose(1,2,0).tolist())),
                         "standardized":standardize(means,counts)}
        if kind in reg["bank"]:
            accept=vetted(reg["bank"][kind],verified_ids)
            summaries[kind]["source_verified"]=accept
            summaries[kind]["with_replay_filter"]={c:means[c] if c[0]=="1" and accept else [0]*4 for c in cells}
            summaries[kind]["without_replay_filter"]={c:means[c] if c[0]=="1" else [0]*4 for c in cells}
    out={"plan_id":reg["id"],"metrics":["net_gain","rescue","harm","extra_tokens"],
         "target_first_opportunity_counts":dict(counts),"reference":"archived natural ascending tasks; unknown cells retained",
         "results":summaries}
    save(root/"analysis.json",out)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("cmd",choices=("collect","report"))
    cli.add_argument("--inputs",type=int,default=3)
    cli.add_argument("--root",type=Path,required=True)
    cli.add_argument("--outputs",type=Path,default=fp.OUTPUT)
    opt=cli.parse_args()
    if opt.cmd=="collect":run(opt.root,opt.outputs,opt.inputs)
    else:report(opt.root,opt.outputs)
