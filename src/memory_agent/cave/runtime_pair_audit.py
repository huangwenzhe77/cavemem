"""Verify runtime sampling, risk sets, frozen predictions, paired artifacts and utility."""
import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from . import formal, formal_plan as fp, gate_validation as gv, prospective
from .branches import revision_fits
from .constrained_gate import Policy
from .experiment import archive_digest
from .gate import lines, signed
from .models import FrozenQwen
from .records import digest, read, save, verify
from .replay import require, restore


def expected_coin(stream, seed, turn):
    key = f"runtime-pair/v1/{stream}/{seed}/{turn}".encode()
    numerator = int(hashlib.sha256(key).hexdigest()[:16], 16) // 2048
    return numerator / (1 << 53)


def audit(root):
    plan = signed(root / "preregistration.json", "plan_id")
    run = root / "blind"
    mf = read(run / "manifest.json")
    rows = lines(run / "index.jsonl")
    source = plan["cohorts"][0]["tasks"]
    require(plan["runtime_code"] == fp.code_digest("runtime_pair.py") and len(rows)==len(source), "protocol or denominator")
    cards = prospective.verified_cards(plan, fp.DEFAULT_BANK, fp.DEFAULT_PACKAGE,
                                       root.parent / "cave-pilot-01/preregistration.json")
    model = Policy(root.parent / "cave-primal-dual-01/model.json", root.parent / "cave-mlp-01/frozen/model.json")
    tokenizer = AutoTokenizer.from_pretrained(plan["actor"]["actor"], local_files_only=True)
    counter = SimpleNamespace(count_prompt_tokens=lambda ms:len(tokenizer(FrozenQwen.prompt(ms)).input_ids))
    seal = read(root / "blind-seal.json")
    require(seal["archive"] == [archive_digest(run,r) for r in rows] and model.doc["policy_id"] ==
            seal["model_id"] == plan["policy_id"], "archive or model drift")
    predictions = []
    reasons = Counter()
    risk_ranks = Counter()
    for entry, r in zip(source, rows, strict=True):
        require(all(r[k] == entry[k] for k in entry) and r["task_digest"]==digest(r["task"]), "task drift")
        selected = None
        rank = 0
        for node in r["steps"]:
            cp = node["checkpoint_id"]
            turn = node["turn_id"]
            pre, draft, _ = restore(run, r, turn)
            actual = read(run / cp / "hazard.json")
            if selected is not None or not node["executable"]:
                reason = "after_first_hit" if selected else "not_executable"
            else:
                d = gv.route(cards,r["task"],pre,draft)
                funded = d.get("rho")==1 and pre["used_tool_calls"]+len(draft["a0"]["commands"]) <= plan["total_tools"] and revision_fits(
                    mf,r,turn,counter,pre,draft,d["reminder"],plan["total_tokens"])
                reason = "eligible" if funded else ("budget" if d.get("rho")==1 else d["status"])
                if funded:
                    rank += 1
                    risk_ranks[rank] += 1
                    coin = expected_coin("selection",r["seed"],turn)
                    require(actual["uniform"]==coin and actual["hit"]==(coin<.5) and actual["hazard"]==.5
                            and actual["survival"]==.5**(rank-1) and actual["first_hit_prob"]==.5**rank,
                            "risk probability drift")
                    if coin<.5:
                        selected = cp
                        x = formal.gate_inputs(r,pre,draft,d,plan)
                        pred = read(run / cp / "prediction.json")
                        require(pred["features"]==x and pred["pre_digest"]==digest(pre) and pred["policy"]==
                                model.decide(x,expected_coin("policy",r["seed"],turn)), "prediction drift")
                        predictions.append(pred)
            require(reason == actual["reason"], "risk set changed")
            reasons[reason] += 1
        require(selected == r["selected"] and rank == r["eligible_draws"], "selection drift")
    require(predictions==lines(root/"predictions.jsonl") and digest(predictions)==seal["predictions_digest"], "prediction seal")
    pairs = read(root/"paired/pairs.json")
    checks = read(root/"paired/checks.json")
    require(len(pairs)==len(predictions) and all(c["tape_status"]=="pass" for c in checks), "pair denominator")
    # Unselected episodes share their measured no-intervention trajectory across all target policies.
    policies = ("silent", "always_remind", "ready_rule", "supervised", "primal_dual")
    totals = {k:Counter() for k in policies}
    for row in rows:
        if row["selected"] is None:
            for acc in totals.values():
                acc.update(tasks=1, success=row["verifier"]["reward"], tokens=row["input_tokens"]+row["output_tokens"])
    for p, pred in zip(pairs,predictions,strict=True):
        if p["status"]!="complete":
            continue
        require(p["checkpoint_id"]==pred["checkpoint_id"] and p["a0_digest"]==pred["a0_digest"],"wrong fork")
        task = next(r["task"] for r in rows if r["episode_id"]==p["episode_id"])
        for name, arm in p["arms"].items():
            raw=read(root/"paired"/arm["file"])
            y=verify(task,raw["state"])["reward"]
            u=raw["usage"]
            sec=sum(u[k] for k in ("common_actor_seconds","new_actor_seconds","new_tool_seconds"))
            utility=y-plan["lambda_tokens"]*(u["input_tokens"]+u["output_tokens"])/1000-plan["lambda_tool"]*u["tool_calls"]-plan["lambda_time"]*sec/60
            require(digest(raw)==arm["digest"] and y==arm["Y"] and math.isclose(utility,arm["utility"])
                    and raw["revision_calls"]==int(name=="remind"), "arm budget or terminal result changed")
        require(math.isclose(p["difference"],p["arms"]["remind"]["utility"]-p["arms"]["silent"]["utility"]), "wrong D")
        decisions = (False, True, pred["ready_rule"], model.base.predict(pred["features"])["remind"], pred["policy"]["remind"])
        silent = p["arms"]["silent"]
        for name, use in zip(policies,decisions,strict=True):
            a = p["arms"]["remind"] if use else silent
            u = a["usage"]
            totals[name].update(tasks=1, success=a["Y"], tokens=u["input_tokens"]+u["output_tokens"],
                                reminders=int(use), gains=a["utility"]-silent["utility"],
                                rescues=int(a["Y"]>silent["Y"]), regressions=int(a["Y"]<silent["Y"]))
    summary={"plan_id":plan["plan_id"],"episodes":len(rows),"reasons":dict(reasons),
             "at_risk_by_rank":dict(risk_ranks),"predictions":len(predictions),"pairs":len(pairs),
             "complete":sum(p["status"]=="complete" for p in pairs),"aa_agree":sum(c.get("free_status")=="agree" for c in checks),
             "avoided_selected":sum(p["e"]==0 for p in predictions),"pairs_digest":digest(pairs),"seal_digest":digest(seal),
             "policy_reconstruction":totals}
    save(root/"audit.json",summary)
    print(summary)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--root",type=Path,required=True)
    audit(cli.parse_args().root)

