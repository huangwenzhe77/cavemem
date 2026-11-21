"""Register and collect disjoint paired-training, logged-training and calibration tasks."""

import argparse
import random
import secrets
from pathlib import Path
from time import perf_counter

from . import formal, formal_plan as fp, gate, gate_validation as gv, prospective, single_arm as sa
from .__main__ import collect_episode
from .branches import branch, revision_fits
from .models import FrozenQwen, xml_parser
from .records import digest, read
from .replay import replay_tape, require, restore
from .supplement_plan import SHAPES, make_input


def register(root, outputs, blocks):
    require(blocks >= 4, "at least four blocks required")
    parent = outputs/"cave-single-arm-01"
    plan, pid = fp.load(parent)
    seen = {tuple(sorted(t["task"]["input"])) for t in read(parent/"sealed-test.json")["tasks"]}
    for path in outputs.glob("cave-*/**/preregistration.json"):
        for c in read(path).get("cohorts", []):
            seen.update(tuple(sorted(r["task"]["input"])) for r in c.get("tasks", [])
                        if isinstance(r, dict) and "input" in r.get("task", {}))
    for path in outputs.glob("cave-*/**/manifest.json"):
        m = read(path)
        if m.get("benchmark") == "mini-data-v1":
            seen.update(tuple(sorted(t["input"])) for t in m["tasks"])
    names = [k for k, v in gv.TEMPLATES.items() if v[0] == "descending"]
    rows, seed = [], 5270000
    key = secrets.token_bytes(32)
    for block in range(blocks):
        name = names[(block//len(SHAPES)) % len(names)]
        shape = SHAPES[block % len(SHAPES)]
        phases = ["paired", "logged", "logged", "calibration"]
        random.Random(f"design:{block}").shuffle(phases)
        for phase in phases:
            while True:
                seed += 1
                values = make_input(seed, shape, "descending")
                if values and not any(fp.near_duplicate(values, s) for s in seen):
                    break
            seen.add(tuple(sorted(values)))
            task = dict(template_id=name, instruction=gv.TEMPLATES[name][1], family="descending", input=values)
            rows.append(dict(episode_id=f"ep-{len(rows):04d}", seed=seed, phase=phase, block=block, shape=shape,
                             task=task, task_digest=digest(task), main_opportunity=1, q=1.))
    doc = {"parent_plan": pid, "seed_hex": key.hex(), "code": fp.code_digest("comparison_cohort.py"),
           "bank_id": plan["bank_id"], "cohorts": [{"split": "development", "tasks": rows}],
           "design": "first executable opportunity; stratified four-task blocks; pair/2 logged/calibration; "
                     "compare identical frozen encoder, warm start and observed-continuation budget"}
    root.mkdir(exist_ok=False)
    sa.durable(root/"preregistration.json", {**doc, "plan_id": digest(doc)})


def priced(plan, y, tokens, tools, seconds):
    require(y in (0,1), "unscored outcome")
    return {"y": y, "tokens": tokens, "tools": tools, "seconds": seconds,
            "r": y-plan["lambda_tokens"]*tokens/1000-plan["lambda_tool"]*tools-plan["lambda_time"]*seconds/60}


def collect(root, outputs):
    reg = gate.signed(root/"preregistration.json", "plan_id")
    plan, pid, _, cards = sa.validate(outputs/"cave-single-arm-01", fp.DEFAULT_BANK, fp.DEFAULT_PACKAGE)
    require(pid == reg["parent_plan"] and reg["code"] == fp.code_digest("comparison_cohort.py"), "plan changed")
    mf = {"actor": plan["actor"], "max_steps": plan["max_steps"], "cohort": "natural"}
    with sa.guard(root/"run.lock"):
        actor = FrozenQwen(plan["actor"]["actor"], plan["actor"]["max_new_tokens"], 0)
        require(actor.info == plan["actor"], "actor drift")
        parser = xml_parser()
        run = root/"raw"
        run.mkdir(exist_ok=True)
        ledger = []
        for row in reg["cohorts"][0]["tasks"]:
            eid, task = row["episode_id"], row["task"]
            folder = run/eid
            output = folder/"result.json"
            result = {k: row[k] for k in ("episode_id","phase","block","task_digest","shape")}
            if output.exists():
                ledger.append(read(output))
                continue
            if folder.exists():
                result["status"] = "interrupted"
            elif row["phase"] == "logged":
                s = sa.Episode(folder, plan, reg["plan_id"], reg["plan_id"], row, actor, parser, cards,
                               bytes.fromhex(reg["seed_hex"])).run()
                sa.durable(folder/"summary.json", s)
                result["status"] = s["main_status"]
                if s["assignment"]:
                    sel = read(run/s["checkpoint_id"]/"selection.json")
                    u = s["usage"]
                    result.update(x=sel["features"], z=s["assignment"]["Z"], b=s["assignment"]["observed_propensity"],
                                  observed=priced(plan, s["verifier"]["reward"], u["total_tokens"], u["tool_calls"],
                                                  u["actor_seconds"]+u["tool_seconds"]+u["matching_seconds"]))
            else:
                item = collect_episode(run, actor, parser, task, eid, row["seed"], plan["max_steps"], 1)
                sa.durable(folder/"blind.json", item)
                result["status"] = "never_reached"
                if item["main_reached"]:
                    turn = item["main_turn_id"]
                    pre, draft, _ = restore(run, item, turn)
                    start = perf_counter()
                    d = gv.route(cards, task, pre, draft)
                    x = formal.gate_inputs(row, pre, draft, d, plan) if d.get("rho") == 1 else None
                    seconds = perf_counter()-start
                    result["status"] = "ineligible"
                    if x and revision_fits(mf,item,turn,actor,pre,draft,d["reminder"],plan["total_tokens"]) and (
                            pre["used_tool_calls"]+len(draft["a0"]["commands"]) <= plan["total_tools"]):
                        result.update(status="paired", x=x, arms={})
                        sa.durable(folder/"selection.json", {"features": x, "pre_digest": digest(pre),
                                                           "draft_digest": digest(draft)})
                        replay_tape(run, mf, item)
                        check = prospective.check_free(folder/"aa", run, mf, item, {"turn": turn}, plan, actor)
                        require(check["free_status"] == "agree", "free continuation mismatch")
                        sequence = ["silent","remind"]
                        random.Random("order:"+eid).shuffle(sequence)
                        for arm in sequence:
                            raw = branch(run,mf,item,turn,actor,arm,d["reminder"],row["seed"],
                                         plan["total_tokens"],plan["total_tools"])
                            sa.durable(folder/(arm+".json"), raw)
                            u = raw["usage"]
                            prefix_time = sum(read(folder/f"{i:03d}"/"generation.json")["seconds"] for i in range(1,turn+1))
                            result["arms"][arm] = priced(plan,raw["score"]["reward"],u["input_tokens"]+u["output_tokens"],
                                u["tool_calls"],seconds+prefix_time+u["new_actor_seconds"]+u["new_tool_seconds"])
            sa.durable(output,result)
            ledger.append(result)
            print(eid, result["phase"], result["status"], flush=True)
        doc = {"plan_id": reg["plan_id"], "rows": ledger, "digest": digest(ledger)}
        if not (root/"dataset.json").exists():
            sa.durable(root/"dataset.json",doc)
        require(read(root/"dataset.json")==doc,"release changed")


if __name__ == "__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("command",choices=("register","collect"))
    cli.add_argument("--root",type=Path,required=True)
    cli.add_argument("--outputs",type=Path,default=fp.OUTPUT)
    cli.add_argument("--blocks",type=int,default=16)
    opt=cli.parse_args()
    if opt.command=="register":
        register(opt.root,opt.outputs,opt.blocks)
    else:
        collect(opt.root,opt.outputs)
