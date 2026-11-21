"""Matched-budget paired and DR policy fitting with disjoint calibration and abstention."""

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from . import formal_plan as fp, gate, gate_mlp, policy_value as pv, single_arm as sa
from .constrained_gate import CONFIG, optimize, sigmoid, vector
from .records import digest, read, save
from .replay import require


def commands(a):
    return [c["keystrokes"].strip() for c in a["commands"]]


def prefix_profile(state, messages):
    observations = [line.split(":",1)[0] for m in messages[2:] if m["role"] == "user"
                    for line in m["content"].splitlines() if ":" in line]
    return {"observations": sorted(set(observations)),
            "intact": Counter(state["working"]) == Counter(state["input"]),
            "ascending": state["working"] == sorted(state["working"])}


def verified_profiles(outputs):
    bank = gate.signed(fp.DEFAULT_BANK, "bank_id")["cards"][0]
    desc = gate.signed(fp.DEFAULT_PACKAGE/"development-v2/card.json", "memory_id")
    profiles = {}
    for card, path, cp in ((bank, outputs/"cave-acceptance/source", bank["checkpoint_id"]),
                           (desc, outputs/"cave-source-expansion-01/source", desc["checkpoint"])):
        require(read(path/"manifest.json")["run_id"] == card["source_run"], "wrong source trace")
        pre, draft = read(path/cp/"pre.json"), read(path/cp/"draft.json")
        a0 = card.get("a0",card.get("original"))
        require(a0 == draft["a0"], "source proposal changed")
        profiles[card["memory_id"]] = {**prefix_profile(pre["state"],pre["messages"]), "ops": commands(a0),
                                      "checkpoint": cp, "source_run": card["source_run"], "pre_digest": digest(pre)}
    return profiles


def shift_features(x, profiles, no_action=False):
    src = profiles[x["memory"]["id"]]
    target = prefix_profile(x["working_state"],x["history_before_proposal"])
    past, now = set(src["observations"]),set(target["observations"])
    a,b = set(src["ops"]),set(commands(x["proposal"]))
    def distance(left,right):
        return len(left ^ right)/max(1,len(left | right))
    # Exposure evidence is categorical: exact contract match or unknown.
    # This flag is distinct from a calibrated probability of downstream harm.
    return [distance(past,now), float(src["intact"] != target["intact"]),
            float(src["ascending"] != target["ascending"]), 0. if no_action else distance(a,b),
            0. if no_action else float(x["evidence"]["exposed"] in (0,1))]


def features(base,x,profiles,variant="full"):
    v = gate_mlp.encode(x,base.model["cards"],variant,base.tokenizer,base.table)
    return np.append(v,shift_features(x,profiles,variant=="no_a0"))


def pseudo_advantages(rows, cards, *, stratum=None):
    scores, diagnostics = [], []
    for i,r in enumerate(rows):
        # No target outcomes in the fold that estimates either nuisance mean.
        fold = r["block"] % 4
        other = [s for s in rows if s["block"] % 4 != fold]
        ready = stratum or (lambda x: gate.ready(x, cards))
        target_ready = ready(r["x"])
        mu = []
        for arm in (0,1):
            group = [s for s in other if s["z"] == arm]
            require(group, "no training-fold support; collect more independent assignments")
            values = np.array([[s["observed"]["r"],s["observed"]["tokens"]] for s in group])
            cell = [j for j,s in enumerate(group) if ready(s["x"])==target_ready]
            mu.append((2*values.mean(axis=0)+values[cell].sum(axis=0))/(2+len(cell)))
        z,b=r["z"],r["b"]
        require(z in (0,1) and 0 < b < 1, "invalid logged probability")
        y = np.array([r["observed"]["r"],r["observed"]["tokens"]])
        score = mu[1]-mu[0]+(1 if z else -1)*(y-mu[z])/b
        scores.append(score)
        diagnostics.append({"episode_id":r["episode_id"],"block_fold":fold,"propensity":b,
                            "mu":np.asarray(mu).tolist(),"pseudo_difference":score.tolist()})
    return np.array(scores), diagnostics


def risk_upper(events,n,alpha=.05):
    """One-sided exact binomial upper bound on a policy-level event probability."""
    if n==0 or events==n:
        return 1.
    lo,hi=0.,1.
    for _ in range(60):
        p=(lo+hi)/2
        cdf=sum(math.comb(n,k)*p**k*(1-p)**(n-k) for k in range(events+1))
        if cdf>alpha: lo=p
        else: hi=p
    return hi


def calibrate(rows,base,margin,risk_budget):
    # Partition by registered independent blocks, before reading their outcomes.
    gain=[r for r in rows if r["block"]%3 != 2]
    risk=[r for r in rows if r["block"]%3 == 2]
    def point(r):
        return base.predict(r["x"])["predicted_D"]
    def measured(r):
        return r["arms"]["remind"]["r"]-r["arms"]["silent"]["r"]
    residuals=sorted(point(r)-measured(r) for r in gain)
    rank=math.ceil(.90*(len(gain)+1))
    offset=residuals[rank-1] if rank <= len(gain) else None
    # Count all possible reminders at accepted states, giving a conservative bound
    # for every stochastic policy supported inside this fixed abstention rule.
    events=sum(bool(offset is not None and point(r)-offset>margin and
                    r["arms"]["silent"]["y"]==1 and r["arms"]["remind"]["y"]==0) for r in risk)
    upper=risk_upper(events,len(risk))
    return {"offset":offset,"prediction_coverage_target":.9,"policy_risk_confidence":.95,
            "gain_tasks":len(gain),"risk_tasks":len(risk),"possible_harms":events,"risk_upper":upper,
            "margin":margin,"risk_budget":risk_budget,"risk_supported":upper<=risk_budget,
            "supported_instructions":sorted({r["x"]["instruction"] for r in gain} &
                                            {r["x"]["instruction"] for r in risk})}


class Runtime:
    def __init__(self,model,base_path):
        self.doc=gate.signed(model,"id")
        self.base=gate_mlp.load_model(base_path)
        require(self.doc["code"] == fp.code_digest("logged_policy_fit.py") and
                self.doc["base_model"]==self.base.model["model_id"], "trained model provenance")
        self.params={k:[np.asarray(w) for w in head["parameters"]] for k,head in self.doc["heads"].items()}

    def decide(self,x,u,head="dr",guarded=True):
        require(0<=u<1 and head in self.params,"invalid gate request")
        if (not x or x["memory"]["id"] not in self.doc["profiles"] or x["evidence"]["source_verified"]!=1
                or x["evidence"]["applicable"]!=1 or x["budget_after_proposal"]["remaining_actor_calls"]<1):
            return {"probability":0.,"remind":False,"reason":"ineligible"}
        cal=self.doc["calibration"]
        lower=None if cal["offset"] is None else self.base.predict(x)["predicted_D"]-cal["offset"]
        reason="sampled"
        if guarded:
            if x["instruction"] not in cal["supported_instructions"]: reason="outside_calibration_domain"
            elif lower is None or lower<=cal["margin"]: reason="insufficient_gain"
            elif not cal["risk_supported"]: reason="insufficient_policy_risk_evidence"
        v=features(self.base,x,self.doc["profiles"],"no_a0" if head=="paired_no_a0" else "full")
        p=float(sigmoid(gate_mlp.forward(v,self.params[head])[1])) if reason=="sampled" else 0.
        return {"probability":p,"remind":u<p,"gain_prediction_lower":lower,"reason":reason}


def fit(root, outputs, dest):
    require(not dest.exists(),"new model path required")
    reg=gate.signed(root/"preregistration.json","plan_id")
    data=read(root/"dataset.json")
    rows=data["rows"]
    require(reg["plan_id"]==data["plan_id"] and digest(rows)==data["digest"] and
            len(set(r["task_digest"] for r in rows))==len(rows),"invalid study")
    registered=reg["cohorts"][0]["tasks"]
    for row, spec in zip(rows,registered,strict=True):
        folder=root/"raw"/spec["episode_id"]
        require(row==read(folder/"result.json") and all(row[k]==spec[k] for k in
                ("episode_id","task_digest","phase","block","shape")) and row["status"]!="interrupted",
                "invalid or interrupted training cohort")
        if row["phase"]=="logged" and row["status"]=="assigned":
            s=read(folder/"summary.json")
            draw=sa.assign(bytes.fromhex(reg["seed_hex"]),reg["plan_id"],row["episode_id"])
            require((draw["Z"],draw["observed_propensity"])==(row["z"],row["b"]) and
                    row["x"]==read(root/"raw"/s["checkpoint_id"]/"selection.json")["features"],
                    "logged decision mismatch")
    base=gate_mlp.load_model(outputs/"cave-mlp-01/frozen/model.json")
    require(set(base.model["task_ids"]).isdisjoint(r["task_digest"] for r in rows),"warm-start leakage")
    profiles=verified_profiles(outputs)
    groups={"paired":[],"logged":[],"calibration":[]}
    for block in sorted({r["block"] for r in rows}):
        block_rows=[r for r in rows if r["block"]==block]
        groups["calibration"].extend(r for r in block_rows if r["phase"]=="calibration" and r["status"]=="paired")
        usable=[r for r in block_rows if r["phase"]!="calibration" and r["status"] in ("paired","assigned")]
        if len(usable)==3:
            for r in usable: groups[r["phase"]].append(r)
    paired,logged=groups["paired"],groups["logged"]
    require(len(paired)>=4 and len(logged)==2*len(paired),"unmatched continuation budgets")
    gains,diagnostics=pseudo_advantages(logged,base.model["cards"])
    limits=base.model["label_spec"]
    heads={}
    for k in ("dr","paired","dr_no_risk","paired_no_a0","paired_no_risk","paired_no_kl"):
        is_dr=k.startswith("dr")
        batch=logged if is_dr else paired
        variant="no_a0" if k=="paired_no_a0" else "full"
        X=np.stack([features(base,r["x"],profiles,variant) for r in batch])
        p0=[a.copy() for a in base.params[variant]]
        p0[0]=np.pad(p0[0],((0,X.shape[1]-len(p0[0])),(0,0)))
        if is_dr:
            gain,extra=gains.T
            # H is unobserved. Its pointwise upper envelope is zero only when the
            # observed outcome rules it out (silent failure or reminder success).
            harm=np.array([int(r["observed"]["y"]!=r["z"]) for r in batch])
        else:
            gain,extra=(np.array([r["arms"]["remind"][f]-r["arms"]["silent"][f] for r in batch])
                        for f in ("r","tokens"))
            harm=np.array([int(r["arms"]["silent"]["y"]==1 and r["arms"]["remind"]["y"]==0) for r in batch])
        lim=limits
        if k.endswith("no_risk"):
            harm=np.zeros_like(harm)
            lim={**limits,"policy_risk_delta":.99999}
        heads[k]=optimize(X,gain,harm,extra,p0,base.model["threshold"],lim,0 if k=="paired_no_kl" else CONFIG["kl"])
    calibration=calibrate(groups["calibration"],base,limits["m_label"],limits["policy_risk_delta"])
    doc={"schema":"matched-policy-comparison/v1","code":fp.code_digest("logged_policy_fit.py"),
         "plan_id":reg["plan_id"],"data_digest":data["digest"],"base_model":base.model["model_id"],
         "profiles":profiles,"heads":heads,"calibration":calibration,"dr_crossfit":diagnostics,
         "groups":{k:[r["episode_id"] for r in v] for k,v in groups.items()},
         "branch_outcome_budget_per_objective":len(logged),"registered_tasks":len(rows),
         "risk_note":"paired uses coupled H; logged uses a conservative upper envelope, never individual harm labels"}
    save(dest,{**doc,"id":digest(doc)})


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    s=cli.add_subparsers(dest="cmd",required=True)
    f=s.add_parser("fit")
    f.add_argument("--root",type=Path,required=True)
    f.add_argument("--outputs",type=Path,default=fp.OUTPUT)
    f.add_argument("--out",type=Path,required=True)
    a=s.add_parser("decide")
    a.add_argument("--model",type=Path,required=True)
    a.add_argument("--base",type=Path,default=fp.OUTPUT/"cave-mlp-01/frozen/model.json")
    a.add_argument("--features",type=Path,required=True)
    a.add_argument("--uniform",type=float,required=True)
    opt=cli.parse_args()
    if opt.cmd=="fit": fit(opt.root,opt.outputs,opt.out)
    else: print(Runtime(opt.model,opt.base).decide(read(opt.features),opt.uniform))
