"""Source-only memory construction and blinded one-checkpoint comparisons on official tasks."""
import argparse
import copy
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parents[1]/"src"))
from memory_agent.cave.records import digest, read, save
from memory_agent.cave.alfworld_memory import paired_contrast
from remote import model_command, teacher_command

REG = read(D/"registration.json")
SYS = """You solve the user's programming task in a Linux terminal. The task instruction specifies the deliverable.
Each turn, output one <action> block containing a Bash batch and ending with </action>.
Work in /app. Commands run in a fresh shell each turn and files persist there. You may use shell heredocs
to write multiline source code. Text before the action is ignored. If your work is complete, put # DONE
on its own line inside the block. A second consecutive completion confirms submission. You have five
turns. Only the public task and your own checks are available. Use commands relevant to the task."""


def parse(text):
    m = re.search(r"<action>(.*?)</action>",text,re.S)
    if m is None:
        m = re.search(r"^# action[ \t]*\r?\n(.*?)^# /action[ \t]*\r?$",text,re.M|re.S)
        if m is None:
            return None, False
    script = m[1].strip()
    script = re.sub(r"^\x60\x60\x60(?:bash|sh)?\s*\n|\n\x60\x60\x60$","",script)
    return script, bool(re.search(r"^\s*#\s*DONE\s*$",script,re.M))


class Worker:
    def __init__(self):
        self.process = subprocess.Popen(model_command(f"{D.name}/infer.py"),
                                        stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding="utf-8")

    def generate(self,messages,seed=None):
        data=json.dumps({"messages":messages,"seed":seed},ensure_ascii=True)
        self.process.stdin.write(data+"\n")
        self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:
            raise RuntimeError("Qwen worker exited")
        ans=json.loads(line)
        action,done=parse(ans["text"])
        return {**ans,"action":action,"done":done,"seed":seed,"prompt_hash":digest(messages)}

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=90)


def reseed(seed, step, purpose):
    if seed is None:
        return None
    return int(digest([seed,step,purpose])[:12],16)


def matches(condition, instruction):
    return bool(re.search({"regex_work":r"regex|regular expression","python_work":r"python|\.py\b|async",
                           "log_aggregation":r"\blogs?\b", "artifact_work":r"file|/app/|write|create|implement"}[condition],
                          instruction,re.I))


def costs(rows, success, extra=None):
    drafts=[r["draft"] for r in rows]
    if extra:
        drafts.append(extra)
    tokens=sum(x["input_tokens"]+x["output_tokens"] for x in drafts)
    sec=sum(x["seconds"] for x in drafts)+sum(r["execution"].get("seconds",0) for r in rows)
    steps=sum(r["draft"]["action"] is not None for r in rows)
    p=REG["cost"]
    return {"success":success,"tokens":tokens,"seconds":sec,"steps":steps,
            "utility":success-p["per_1000_tokens"]*tokens/1000-p["per_env_step"]*steps-p["per_minute"]*sec/60}


def rollout(bk, worker, task, dest, *, natural=None, turn=1, arm="silent", candidate=None, seed=None):
    result=dest/"result.json"
    if result.exists():
        return read(result)
    dest.mkdir(parents=True,exist_ok=True)
    instruction=bk.instruction(task)
    if natural is None:
        app=bk.new(task)
        messages=[{"role":"system","content":SYS},{"role":"user","content":instruction}]
        rows=[]
        pending=False
        original=None
    else:
        pre=read(natural/f"pre-{turn}.json")
        app=bk.clone(task,pre["snapshot"])
        if bk.state(task,app)!=pre["state"]:
            raise ValueError("checkpoint clone state mismatch")
        messages=pre["messages"]
        pending=pre["pending"]
        rows=read(natural/"result.json")["rows"][:turn-1]
        original=pre["draft"]
        if digest(messages)!=original["prompt_hash"]:
            raise ValueError("checkpoint prompt mismatch")
    extra=None
    for step in range(turn,REG["turns"]+1):
        if step==turn and original is not None:
            draft=original
            if arm=="direct":
                draft={"text":"<action>"+candidate["correction"]+"</action>", "action":candidate["correction"],
                       "done":original["done"],"input_tokens":0,"output_tokens":0,"seconds":0.,"seed":None}
                extra=original
            elif arm=="remind":
                revised=copy.deepcopy(messages)
                revised[-1]["content"]+=("\nYour cached proposal was:\n"+original["text"]+
                    "\nAdvisory memory from an earlier task: "+candidate["reminder"]+
                    "\nChoose your action for this step.")
                draft=worker.generate(revised,reseed(seed,step,"revision"))
                extra=original
        elif natural is None and step==1 and (D/"initial-drafts"/(task+".json")).exists():
            draft=read(D/"initial-drafts"/(task+".json"))
            if draft["prompt_hash"]!=digest(messages):
                raise ValueError("cached initial draft prompt differs")
            draft["action"],draft["done"]=parse(draft["text"])
        else:
            draft=worker.generate(messages,reseed(seed,step,"continuation"))
        if natural is None:
            snap=bk.clone(task,app)
            save(dest/f"pre-{step}.json",{"snapshot":snap,"state":bk.state(task,snap),"messages":messages,
                                         "draft":draft,"pending":pending,"step":step})
        if draft["action"] is None:
            obs={"code":None,"stdout":"","stderr":"Invalid action format. Provide a complete <action> block.","seconds":0.}
        else:
            obs=bk.execute(task,app,draft["action"])
        rows.append({"step":step,"draft":draft,"execution":obs})
        save(dest/f"step-{step}.json",rows[-1])
        if draft["done"] and pending:
            break
        pending=draft["done"]
        observation=f"Shell exit: {obs['code']}\n"+obs["stdout"][-4000:]+obs["stderr"][-1000:]
        if pending:
            observation+="\nYou indicated completion. Confirm with # DONE only if the requested deliverable is ready."
        messages=[*messages,{"role":"assistant","content":draft["text"]},{"role":"user","content":observation}]
    seal={"state":bk.state(task,app),"rows_digest":digest(rows),"model":REG["actor"]}
    save(dest/"seal.json",seal)
    verdict=bk.grade(task,app,dest/"verifier")
    if verdict.get("reward") not in (0,1):
        raise RuntimeError(f"official verdict unavailable: {task}, {dest}")
    out={"task":task,"arm":arm,"rows":rows,"seed":seed,"turn":turn,
         "metrics":costs(rows,int(verdict["reward"]),extra),"seal":seal,"verdict":verdict}
    save(result,out)
    print(task,dest.relative_to(D),out["metrics"],flush=True)
    return out


def teacher(task, natural):
    dest=D/"source"/task/"proposal.json"
    if dest.exists():
        return read(dest)["candidate"]
    import backend as bk
    source={"task":bk.instruction(task),"steps":[{"step":r["step"],"action":r["draft"]["action"],
            "done":r["draft"]["done"],"public_feedback":r["execution"]} for r in natural["rows"]],
            "success":natural["metrics"]["success"]}
    references=[]
    for peer in REG["splits"]["source"]:
        peer_path=D/"natural"/peer/"result.json"
        if peer!=task and peer_path.exists():
            trace=read(peer_path)
            if trace["metrics"]["success"]:
                references.append({"task":bk.instruction(peer),"actions":[r["draft"]["action"] for r in trace["rows"]]})
    source["successful_source_examples"]=references
    p=subprocess.run(teacher_command(f"{D.name}/teacher.py"),input=json.dumps(source),
                     capture_output=True,text=True,encoding="utf-8",timeout=300)
    if p.returncode:
        raise RuntimeError("teacher failed: "+p.stderr[-800:])
    out=json.loads(p.stdout)
    save(dest,out)
    return out["candidate"]


def verify(bk,w,task,n):
    folder=D/"source"/task
    c=teacher(task,n)
    if not matches(c["condition"],bk.instruction(task)):
        save(folder/"evidence.json",{"admitted":False,"reason":"condition misses source","card":c})
        return None
    reps=[]
    for seed in REG["seeds"]:
        metrics={}
        order=["silent","direct","remind"]
        random.Random(seed).shuffle(order)
        for arm in order:
            row=rollout(bk,w,task,folder/str(seed)/arm,natural=D/"natural"/task,turn=c["step"],
                        arm=arm,candidate=c,seed=seed)
            metrics[arm]=row["metrics"]
        reps.append({"seed":seed,"metrics":metrics})
    q={a:paired_contrast(reps,a) for a in ("direct","remind")}
    admitted=all(x["supports_gain"] for x in q.values())
    evidence={"admitted":admitted,"source_task":task,"q":q,"replicates":reps,"card":c}
    save(folder/"evidence.json",evidence)
    return {**c,"source":task,"evidence_digest":digest(evidence)} if admitted else None


def target(bk,w,task,bank):
    folder=D/"natural"/task
    natural=read(folder/"result.json")
    m=next((c for c in bank if matches(c["condition"],bk.instruction(task))),None)
    if not m:
        return {"task":task,"matched":False,"silent":natural["metrics"]}
    # Every candidate uses public instruction predicates, so the first eligible
    # checkpoint is the first executable natural proposal. Its direction is unrestricted.
    eligible=next((r for r in natural["rows"] if r["draft"]["action"] is not None),None)
    if eligible is None:
        return {"task":task,"matched":False,"reason":"no executable draft","silent":natural["metrics"]}
    step=eligible["step"]
    cached=read(folder/f"pre-{step}.json")
    source_pre=read(D/"natural"/m["source"]/f"pre-{m['step']}.json")
    action=eligible["draft"]["action"].strip()
    known_bad=source_pre["draft"]["action"]
    exposed=(0 if action==m["correction"].strip() else
             1 if known_bad and action==known_bad.strip() else None)
    decision={"step":step,"original_action":action,"state":cached["state"],"v":1,"rho":1,
              "e":exposed,"basis":"exact correction / source failure command match; otherwise unknown",
              "card":digest(m),"source_evidence":m["evidence_digest"]}
    save(D/"target"/task/"decision.json",decision)
    arms={}
    for a in sorted(("silent","remind"),key=lambda a:digest([task,step,a])):
        row=rollout(bk,w,task,D/"target"/task/a,natural=folder,turn=step,arm=a,candidate=m)
        arms[a]=row["metrics"]
        if a=="silent" and (row["seal"]["state"]!=natural["seal"]["state"] or
                            row["metrics"]["success"]!=natural["metrics"]["success"]):
            raise ValueError("greedy silent A/A did not reproduce natural run")
    return {"task":task,"matched":True,"step":step,"source":m["source"],"exposure":exposed,**arms,
            "delta":{k:arms["remind"][k]-arms["silent"][k] for k in arms["silent"]}}


def run(stage, tasks):
    w=Worker()
    try:
        if stage=="drafts":
            for task in tasks:
                path=D/"initial-drafts"/(task+".json")
                if path.exists():
                    continue
                instruction=(D/"assets"/task/"instruction.md").read_text(encoding="utf-8")
                draft=w.generate([{"role":"system","content":SYS},{"role":"user","content":instruction}])
                save(path,draft)
                print("draft cached",task,draft["output_tokens"],flush=True)
            return
        import backend as bk
        if stage=="natural":
            for task in tasks:
                rollout(bk,w,task,D/"natural"/task)
            return
        if stage=="source":
            for task in tasks:
                if task not in REG["splits"]["source"]:
                    raise ValueError("only historical games eligible")
                n=read(D/"natural"/task/"result.json")
                if n["metrics"]["success"]==0:
                    verify(bk,w,task,n)
            return
        bank=[]
        for task in REG["splits"]["source"]:
            n=read(D/"natural"/task/"result.json")
            if n["metrics"]["success"]==0:
                e=read(D/"source"/task/"evidence.json")
                if e["admitted"]:
                    bank.append({**e["card"],"source":task,"evidence_digest":digest(e)})
        save(D/"bank.json",bank)
        rows={}
        for split in ("train","validation","test"):
            if split=="test":
                save(D/"frozen-gate.json",{"policy":"silent","reason":"fewer than four registered training tasks",
                                            "bank":digest(bank),"development":digest(rows)})
            rows[split]=[target(bk,w,task,bank) for task in REG["splits"][split]]
        save(D/"summary.json",{"registration":digest(REG),"bank_size":len(bank),"targets":rows,
             "source":{t:read(D/"natural"/t/"result.json")["metrics"] for t in REG["splits"]["source"]}})
    finally:
        w.close()


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=["drafts","natural","source","target"])
    ap.add_argument("tasks",nargs="*")
    a=ap.parse_args()
    run(a.stage,a.tasks)
