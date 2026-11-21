"""Prospective first-hit sampling of eligible live drafts, followed by paired replay."""
import argparse
import hashlib
import random
import time
import uuid
from collections import Counter
from pathlib import Path

from . import formal, formal_plan as fp, gate_validation as gv, mini, prospective
from .branches import load_actor, revision_fits
from .constrained_gate import Policy
from .experiment import archive_digest
from .gate import lines, ready, signed
from .models import SYSTEM, xml_parser
from .records import digest, read, save, verify
from .replay import advance, parse, replay_tape, require, restore
from .supplement_plan import SHAPES, make_input


def uniform(stream, seed, turn):
    b = hashlib.sha256(f"runtime-pair/v1/{stream}/{seed}/{turn}".encode()).digest()
    return (int.from_bytes(b[:8], "big") >> 11) / 2**53


def register(root):
    old = root.parent / "cave-single-arm-01"
    p, parent = fp.load(old)
    seen = {tuple(sorted(t["task"]["input"])) for t in read(old / "sealed-test.json")["tasks"]}
    for path in root.parent.glob("cave-*/**/manifest.json"):
        m = read(path)
        if m.get("benchmark") == "mini-data-v1":
            seen.update(tuple(sorted(t["input"])) for t in m["tasks"])
    for c in p["cohorts"]:
        seen.update(tuple(sorted(t["task"]["input"])) for t in c["tasks"])
    inspect_plan = root.parent / "cave-inspect-01/preregistration.json"
    if inspect_plan.exists():
        for c in read(inspect_plan).get("cohorts", []):
            seen.update(tuple(sorted(t["task"]["input"])) for t in c["tasks"])
    prior = digest(sorted(seen))
    rows = []
    for i, (name, (family, text)) in enumerate(gv.TEMPLATES.items()):
        seed = 2110000 + 10000*i
        for n in range(8):
            shape = SHAPES[n % len(SHAPES)]
            while True:
                seed += 1
                values = make_input(seed, shape, family)
                if values is not None and not any(fp.near_duplicate(values, t) for t in seen):
                    break
            seen.add(tuple(sorted(values)))
            task = dict(instruction=text, template_id=name, family=family, input=values)
            rows.append(dict(episode_id=f"ep-{len(rows):04d}", seed=seed, task=task,
                             task_digest=digest(task), input_stratum=shape))
    model = signed(root.parent/"cave-primal-dual-01/model.json", "policy_id")
    plan = {**p, "schema":"runtime-pair/v1", "parent_plan_id":parent, "hazard":.5,
            "prior_inputs_digest":prior, "policy_id":model["policy_id"], "runtime_code":fp.code_digest("runtime_pair.py"),
            "cohorts":[{"split":"development","tasks":rows}], "split_counts":{"development":len(rows)},
            "design":"First hit on live applicable and funded executable drafts; pair both arms after prediction seal."}
    for key in ("assignment_probabilities","outcome_contract"):
        plan.pop(key,None)
    root.mkdir(exist_ok=False)
    save(root/"preregistration.json", {**plan, "plan_id":digest(plan)})
    return signed(root/"preregistration.json", "plan_id")


def episode(run, row, mf, plan, actor, parser, cards, policy):
    spec, eid, seed = row["task"], row["episode_id"], row["seed"]
    state = mini.initial(spec)
    msg = [{"role":"system","content":SYSTEM},{"role":"user","content":spec["instruction"]+
           "\nCurrent working list: "+str(state["working"])}]
    n_in = n_out = tools = opportunities = risk = 0
    steps, selected = [], None
    end = "budget"
    save(run/eid/"sampling.json", {"hazard":.5,"seed":seed})
    for turn in range(1,plan["max_steps"]+1):
        cp = f"{eid}/{turn:03d}"
        folder = run/cp
        seed_t = random.Random(f"actor:{seed}:{turn}").randrange(2**31)
        pre = dict(checkpoint_id=cp, step=turn, messages=msg, state=state, decode_seed=seed_t,
                   remaining_steps=plan["max_steps"]-turn+1, used_input_tokens=n_in, used_output_tokens=n_out,
                   used_tool_calls=tools, memory_enabled=False, teacher_enabled=False)
        save(folder/"pre.json", pre)
        gen = actor.generate(msg, seed_t)
        save(folder/"generation.json",gen)
        action, error = parse(gen["text"], parser)
        msg = msg+[{"role":"assistant","content":gen["text"]}]
        n_in += gen["prompt_tokens"]
        n_out += gen["completion_tokens"]
        draft = dict(checkpoint_id=cp,messages=msg,state=state,a0=action,a0_digest=digest(action),error=error,
                     warning=parser.parse_response(gen["text"]).warning,remaining_steps=pre["remaining_steps"]-1,
                     used_input_tokens=n_in,used_output_tokens=n_out,used_tool_calls=tools)
        save(folder/"draft.json",draft)
        executable = bool(action["commands"]) and not error
        opportunities += int(bool(executable))
        trial = {"checkpoint_id":cp,"hazard":0.,"reason":"after_first_hit" if selected else "not_executable"}
        if executable and selected is None:
            start = time.perf_counter()
            d = gv.route(cards,spec,pre,draft)
            funded = d.get("rho")==1 and tools+len(action["commands"])<=plan["total_tools"] and revision_fits(
                mf,row,turn,actor,pre,draft,d["reminder"],plan["total_tokens"])
            trial.update(reason="eligible" if funded else ("budget" if d.get("rho")==1 else d["status"]),
                         memory_id=d.get("memory_id"),rho=d.get("rho"),e=d.get("e"))
            if funded:
                risk += 1
                u = uniform("selection",seed,turn)
                trial.update(hazard=.5,eligible_rank=risk,survival=.5**(risk-1),first_hit_prob=.5**risk,
                             uniform=u,hit=u<.5)
                if u<.5:
                    x = formal.gate_inputs(row,pre,draft,d,plan)
                    pred = policy.decide(x,uniform("policy",seed,turn))
                    selected = dict(**d,turn=turn,checkpoint_id=cp,episode_id=eid,a0_digest=draft["a0_digest"],
                                    pre_digest=digest(pre),features=x,policy=pred,
                                    ready_rule=not ready(x,policy.base.model["cards"]))
                    save(folder/"prediction.json",selected)
            trial["matching_and_selection_seconds"] = time.perf_counter()-start
        save(folder/"hazard.json",trial)
        obs, executed, stop = advance(state, action, error)
        tools += len(executed["commands"]) if executed else 0
        save(folder/"execution.json",dict(executed_action=executed,observation=obs,state=state,used_tool_calls=tools))
        steps.append(dict(step=turn,turn_id=turn,checkpoint_id=cp,executable=bool(executable),
                          opportunity_index=opportunities if executable else None))
        if stop:
            end=stop
            break
        msg=msg+[{"role":"user","content":"\n".join(obs)}]
    score=verify(spec,state)
    save(run/eid/"verdict.json",score)
    return {**row,"run_id":mf["run_id"],"split":mf["split"],"steps":steps,"termination":end,"verifier":score,
            "input_tokens":n_in,"output_tokens":n_out,"tool_calls":tools,"eligible_draws":risk,
            "selected":selected["checkpoint_id"] if selected else None}


def run_all(root):
    plan = register(root)
    run = root/"blind"
    rows = plan["cohorts"][0]["tasks"]
    mf = dict(schema=1,run_id=uuid.uuid4().hex,benchmark="mini-data-v1",split="development",
              actor=plan["actor"],tasks=[r["task"] for r in rows],max_steps=plan["max_steps"],cohort="natural",
              memory_enabled=False,teacher_enabled=False)
    save(run/"manifest.json",mf)
    actor = load_actor(mf)
    cards = prospective.verified_cards(plan,fp.DEFAULT_BANK,fp.DEFAULT_PACKAGE,root.parent/"cave-pilot-01/preregistration.json")
    p = Policy(root.parent/"cave-primal-dual-01/model.json",root.parent/"cave-mlp-01/frozen/model.json")
    require(p.doc["policy_id"]==plan["policy_id"], "policy drift")
    parser = xml_parser()
    with (run/"index.jsonl").open("x",encoding="utf-8") as idx:
        from .records import encode
        for row in rows:
            item = episode(run,row,mf,plan,actor,parser,cards,p)
            idx.write(encode(item)+"\n")
            idx.flush()
            print("blind",item["episode_id"],item["selected"],flush=True)
    items = lines(run/"index.jsonl")
    predictions=[read(run/r["selected"]/"prediction.json") for r in items if r["selected"]]
    formal.write_rows(root/"predictions.jsonl",predictions)
    seal=dict(plan_id=plan["plan_id"],model_id=p.doc["policy_id"],archive=[archive_digest(run,r) for r in items],
              predictions_digest=digest(predictions))
    save(root/"blind-seal.json",seal)
    return pair(root,actor)


def pair(root, actor=None):
    plan = signed(root/"preregistration.json","plan_id")
    run = root/"blind"
    mf, items = read(run/"manifest.json"), lines(run/"index.jsonl")
    seal=read(root/"blind-seal.json")
    require(seal["archive"]==[archive_digest(run,r) for r in items] and seal["plan_id"]==plan["plan_id"]
            and seal["predictions_digest"]==digest(lines(root/"predictions.jsonl")), "sealed blind inputs changed")
    prefix=root/"paired"
    prefix.mkdir(exist_ok=False)
    actor=actor or load_actor(mf)
    decisions, checks=[],[]
    for r in items:
        replay_tape(run,mf,r)
        cp=r["selected"]
        check={"episode_id":r["episode_id"],"tape_status":"pass","archive_digest":archive_digest(run,r)}
        d=read(run/cp/"prediction.json") if cp else {}
        if cp:
            check.update(prospective.check_free(prefix/"aa"/r["episode_id"],run,mf,r,d,plan,actor))
        decisions.append(d)
        checks.append(check)
    save(prefix/"checks.json",checks)
    pairs=prospective.collect_pairs(prefix,run,mf,items,decisions,checks,plan,seal,actor)
    cells=Counter()
    gain=Counter()
    for result in pairs:
        if result["status"]!="complete":
            cells[result["status"]]+=1
            continue
        arms=result["arms"]
        for arm in arms.values():
            require(arm["digest"]==digest(read(prefix/arm["file"])), "arm changed")
        s,r=arms["silent"],arms["remind"]
        cell={(1,1):"both_success",(1,0):"regression",(0,1):"rescue",(0,0):"both_fail"}[s["Y"],r["Y"]]
        cells[cell]+=1
        ts=lambda a: a["usage"]["input_tokens"]+a["usage"]["output_tokens"]
        if cell=="both_success" and ts(r)>ts(s):
            cells["both_success_more_tokens"]+=1
        pred=read(run/result["checkpoint_id"]/"prediction.json")
        for label,on in [("policy",pred["policy"]["remind"]),("rule",pred["ready_rule"]),("always_remind",True)]:
            if on:
                gain[label+"_reminders"]+=1
                gain[label+"_extra_tokens"]+=ts(r)-ts(s)
                gain[label+"_gain"]+=result["difference"]
                gain[label+"_success_delta"]+=r["Y"]-s["Y"]
    out={"plan_id":plan["plan_id"], "tasks":len(items),"risk_set_tasks":sum(r["eligible_draws"]>0 for r in items),
         "eligible_draws":sum(r["eligible_draws"] for r in items),"selected":len(pairs),"cells":dict(cells),
         "policy_differences_summed_vs_silent":dict(gain),"baseline_success":sum(r["verifier"]["reward"] for r in items),
         "no_hit_tasks":[r["episode_id"] for r in items if r["selected"] is None]}
    save(root/"summary.json",out)
    print(out,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    run_all(ap.parse_args().root)

