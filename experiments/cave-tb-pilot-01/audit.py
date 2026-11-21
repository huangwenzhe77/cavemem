"""Recompute the pilot report from sealed trajectories and official score receipts."""
import json
import hashlib
from collections import Counter
from pathlib import Path
from statistics import mean

from experiment import D, REG, costs, digest, read, save


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def check(folder):
    run=read(folder/"result.json")
    receipt=run["verdict"]
    seal=read(folder/"seal.json")
    assert run["seal"]==seal and seal["rows_digest"]==digest(run["rows"])
    assert seal["state"]==receipt["state"]
    assert receipt==read(folder/"verifier"/"result.json")
    report=folder/"verifier/verifier/ctrf.json"
    assert sha(report)==receipt["ctrf_sha256"]
    tests=read(report)["results"]["tests"]
    assert tests and all(t["status"] in ("passed","failed") for t in tests)
    assert int(all(t["status"]=="passed" for t in tests))==receipt["reward"]==run["metrics"]["success"]
    assert sha(D/"assets"/run["task"]/"tests/test_outputs.py")==receipt["test_sha256"]
    extra=None
    if run["arm"] in ("direct","remind"):
        extra=read(D/"natural"/run["task"]/f"pre-{run['turn']}.json")["draft"]
    assert costs(run["rows"],receipt["reward"],extra)==run["metrics"]
    return run


def audit():
    s=read(D/"summary.json")
    assert s["registration"]==digest(REG)
    bank=read(D/"bank.json")
    assert read(D/"frozen-gate.json")["bank"]==digest(bank)
    natural={t:check(D/"natural"/t) for names in REG["splits"].values() for t in names}
    source=[]
    for t in REG["splits"]["source"]:
        n=natural[t]
        if n["metrics"]["success"]:
            continue
        e=read(D/"source"/t/"evidence.json")
        r={"task":t,"admitted":e["admitted"],"condition":e["card"]["condition"],
           "step":e["card"]["step"],"reason":e.get("reason")}
        if "replicates" in e:
            r["gains"]=e["q"]
            for rep in e["replicates"]:
                for arm in ("silent","direct","remind"):
                    run=check(D/"source"/t/str(rep["seed"])/arm)
                    assert run["metrics"]==rep["metrics"][arm]
            r["arm_wins"]={a:sum(x["metrics"][a]["success"] for x in e["replicates"])
                           for a in ("silent","direct","remind")}
            r["text_rescues"]=sum(x["metrics"]["silent"]["success"]==0 and
                                  x["metrics"]["remind"]["success"]==1 for x in e["replicates"])
            r["text_dual_success"]=sum(x["metrics"]["silent"]["success"]==1 and
                                       x["metrics"]["remind"]["success"]==1 for x in e["replicates"])
            r["arm_means"]={a:{k:mean(x["metrics"][a][k] for x in e["replicates"])
                               for k in ("tokens","seconds","utility")}
                            for a in ("silent","direct","remind")}
        source.append(r)
    assert {r["task"] for r in source if r["admitted"]}=={c["source"] for c in bank}
    pairs=[]
    groups={}
    for split, rows in s["targets"].items():
        assert [r["task"] for r in rows]==REG["splits"][split]
        groups[split]={"games":len(rows),"silent_wins":sum(r["silent"]["success"] for r in rows),
                       "pairs":sum(r["matched"] for r in rows)}
        for r in rows:
            if r["matched"]:
                arms={a:check(D/"target"/r["task"]/a)["metrics"] for a in ("silent","remind")}
                assert all(arms[a]==r[a] for a in arms)
                pairs.append({"split":split,**r,"outcome":{(0,0):"both_fail",(0,1):"rescue",
                         (1,0):"harm",(1,1):"both_success"}[(arms["silent"]["success"],arms["remind"]["success"])],
                         "label":"remind" if r["delta"]["utility"]>0 else "silent"})
    outcomes=Counter(p["outcome"] for p in pairs)
    receipt={"benchmark":"Terminal-Bench 2.0 six-task pilot","official_task_revision":REG["commit"],
             "natural":{t:{"split":next(k for k,l in REG["splits"].items() if t in l),
                 **r["metrics"],"passed_checks":r["verdict"]["summary"]["passed"],
                 "total_checks":r["verdict"]["summary"]["tests"],
                 "parsed_turns":sum(z["draft"]["action"] is not None for z in r["rows"]),
                 "turns":len(r["rows"])} for t,r in natural.items()},
             "sources":source,"verified_cards":len(bank),"groups":groups,"pairs":pairs,
             "outcomes":dict(outcomes) if pairs else None,"gate":read(D/"frozen-gate.json")}
    save(D/"audited-results.json",receipt)
    return receipt


if __name__=="__main__":
    print(json.dumps(audit(),indent=2))
