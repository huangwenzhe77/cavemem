"""Score matched-budget models on archived independent tasks, including no-hit tasks."""

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from . import formal_plan as fp, gate
from .logged_policy_fit import Runtime, shift_features
from .records import digest, read, save
from .replay import require


def evaluate(outputs, study, dest):
    require(not dest.exists(),"new evaluation path required")
    model=Runtime(study/"model.json",outputs/"cave-mlp-01/frozen/model.json")
    reg=gate.signed(study/"preregistration.json","plan_id")
    root=outputs/"cave-runtime-pair-01"
    trials=gate.lines(root/"blind/index.jsonl")
    truth=read(root/"paired/pairs.json")
    pred=gate.lines(root/"predictions.jsonl")
    seal=read(root/"blind-seal.json")
    audit=read(root/"audit.json")
    require(audit["pairs_digest"]==digest(truth) and audit["seal_digest"]==digest(seal) and
            seal["predictions_digest"]==digest(pred),"development archive integrity")
    train={r["task_digest"] for c in reg["cohorts"] for r in c["tasks"]}
    require(train.isdisjoint(digest(r["task"]) for r in trials),"train/calibration and evaluation overlap")
    pairs={r["episode_id"]:r for r in truth}
    xs={p["episode_id"]:p["features"] for p in pred}
    require(set(xs)==set(pairs) and all(r["status"]=="complete" for r in truth),"incomplete pairs")
    names=["silent","always","ready","exposure_only","source_only","supervised",*model.params,"guarded_dr","guarded_paired"]
    mat=np.zeros((len(trials),len(names),7))
    costs=[]
    audit_rows=[]
    price=gate.signed(root/"preregistration.json","plan_id")["lambda_time"]/60
    for i,t in enumerate(trials):
        eid=t["episode_id"]
        if eid not in pairs:
            require(t["verifier"]["reward"] in (0,1),"unknown no-hit outcome")
            mat[i,:,0]=t["verifier"]["reward"]
            mat[i,:,1]=t["input_tokens"]+t["output_tokens"]
            mat[i,:,6]=t["tool_calls"]
            continue
        x,p=xs[eid],pairs[eid]
        arms=p["arms"]["silent"],p["arms"]["remind"]
        require(all(digest(read(root/"paired"/a["file"]))==a["digest"] for a in arms),"arm digest")
        tokens=[a["usage"]["input_tokens"]+a["usage"]["output_tokens"] for a in arms]
        tools=[a["usage"]["tool_calls"] for a in arms]
        y0,y1=[a["Y"] for a in arms]
        decisions={}
        for j,name in enumerate(names):
            def decide():
                if name=="silent":return 0.,None
                if name=="always":return 1.,None
                if name=="ready":return float(not gate.ready(x,model.base.model["cards"])),None
                if name=="source_only":return float(x["evidence"]["source_verified"]==1),None
                if name=="exposure_only":return float(x["evidence"]["exposed"]==1),None
                if name=="supervised":return float(model.base.predict(x)["remind"]),None
                guarded=name.startswith("guarded_")
                a=model.decide(x,.5,name.removeprefix("guarded_"),guarded)
                return a["probability"],a["reason"]
            prob,reason=decide()
            start=perf_counter()
            for _ in range(9):decide()
            elapsed=(perf_counter()-start)/9
            mat[i,j]=[y0+prob*(y1-y0),tokens[0]+prob*(tokens[1]-tokens[0]),
                      prob*p["difference"]-price*elapsed,prob*int(y1<y0),prob*int(y1>y0),prob,
                      tools[0]+prob*(tools[1]-tools[0])]
            decisions[name]={"p":prob,"reason":reason,"gate_seconds":elapsed}
        costs.append({"exposure":x["evidence"]["exposed"],"gain_error":model.base.predict(x)["predicted_D"]-p["difference"]})
        audit_rows.append({"episode_id":eid,"decisions":decisions,"exposure":x["evidence"]["exposed"],
                           "shift":shift_features(x,model.doc["profiles"])})
    metrics=["success","tokens","gain_vs_silent","regressions","rescues","reminders","tools"]
    n=len(trials)
    idx=np.random.default_rng(8419).integers(0,n,(1000,n))
    totals=mat.sum(axis=0)
    d=mat-mat[:,0:1,:]
    b=d[idx].mean(axis=1)
    stats={}
    for j,name in enumerate(names):
        safe=[r["decisions"][name]["p"] for r in audit_rows if r["exposure"]==0]
        stats[name]={"expected_totals":dict(zip(metrics,totals[j].tolist())),
                     "task_mean_delta_ci95":dict(zip(metrics,np.quantile(b[:,j,:],[.025,.975],axis=0).T.tolist())),
                     "p_remind_given_avoided":float(np.mean(safe)) if safe else None}
    probe=read(outputs/"cave-inspect-01/analysis.json")
    report={"model_id":model.doc["id"],"scope":"development archives; policy probabilities integrated over paired outcomes",
            "code":fp.code_digest("comparison_eval.py"),"tasks":n,"pairs":len(pairs),"policies":stats,"decisions":audit_rows,
            "gain_mae_when_exposure_unknown":float(np.mean([abs(c["gain_error"]) for c in costs if c["exposure"] is None]))
            if any(c["exposure"] is None for c in costs) else None,
            "probe_budget_0_vs_1":{"source_seal":probe["seal_digest"],"uncertain":probe["uncertain"],
                                  "eligible":probe["eligible"],"one_probe_supported":probe["inspection_recommended"]}}
    save(dest,report)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--outputs",type=Path,default=fp.OUTPUT)
    cli.add_argument("--study",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    args=cli.parse_args()
    evaluate(args.outputs,args.study,args.out)
