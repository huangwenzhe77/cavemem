"""Seal predictions before a prospective fork and score each policy from verified arms."""

import argparse
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import formal, formal_plan as fp, gate, gate_mlp, gate_validation, prospective
from .gate_eval import score
from .records import digest, read, save, verify
from .replay import require, restore


POLICIES = ("silent", "always_remind", "ready_rule", "stump", "mlp", "mlp_no_a0")


def decide(name, features, stump, mlp):
    if name == "silent":
        return {"remind": False}
    if name == "always_remind":
        return {"remind": True}
    if name == "ready_rule":
        return {"remind": not gate.ready(features, stump["cards"])}
    if name == "stump":
        return gate.predict(stump, features)
    return mlp.predict(features, variant="full" if name == "mlp" else "no_a0")


def predict(args):
    root, out = args.root, args.out
    plan, pid = fp.load(root)
    is_fresh = "trial_registration" in plan
    require(not is_fresh or not (root/"audit-validation").exists(), "late prospective predictions")
    require(not out.exists(), "predictions already sealed")
    mf, rows = fp.checked_run(plan, "validation", root/"validation")
    stump = gate.load_model(args.stump)
    start = time.perf_counter()
    mlp = gate_mlp.load_model(args.model)
    setup = time.perf_counter() - start
    require(stump["bank_id"] == plan["bank_id"] and
            not set(stump["task_ids"]).intersection(digest(r["task"]) for r in rows), "validation overlap or bank drift")
    cards = prospective.verified_cards(plan, fp.DEFAULT_BANK, fp.DEFAULT_PACKAGE,
                                       fp.OUTPUT/"cave-pilot-01/preregistration.json")
    route = gate_validation.route if "validation_templates" in plan else fp.formal_route
    matches, _ = prospective.schedule(root/"validation", mf, rows, cards, route)
    decisions = []
    for row, match in zip(rows, matches, strict=True):
        entry = {"episode_id": row["episode_id"], "match": match["status"], "predictions": {},
                 "gate_seconds": {}, "checkpoint": match.get("checkpoint_id")}
        if match.get("rho") == 1 and match["turn"] < plan["max_steps"]:
            pre, draft, _ = restore(root/"validation", row, match["turn"])
            times = Counter()
            for repetition in range(22):
                order = list(POLICIES)
                random.Random(f"mlp-gate:{row['seed']}:{repetition}").shuffle(order)
                for name in order:
                    start = time.perf_counter_ns()
                    x = formal.gate_inputs(row, pre, draft, match, plan)
                    pred = decide(name, x, stump, mlp)
                    elapsed = (time.perf_counter_ns()-start)/1e9
                    if repetition == 0:
                        entry["predictions"][name] = pred
                    else:
                        require(entry["predictions"][name] == pred, "unstable inference")
                        times[name] += elapsed
            entry.update(gate_seconds={k:v/21 for k,v in times.items()}, features_digest=digest(x))
        decisions.append(entry)
    meta = {"plan_id": pid, "blind_seal": digest(read(root/"validation/blind-seal.json")),
            "model_digest": digest(read(args.model)), "stump_id": stump["model_id"],
            "protocol": fp.code_digest("mlp_evaluation.py"), "prospective": is_fresh,
            "load_seconds": setup, "rows": decisions}
    out.mkdir(parents=True, exist_ok=False)
    save(out/"predictions.json", {**meta, "id": digest(meta)})


def evaluate(args):
    root, out = args.root, args.out
    plan, pid = fp.load(root)
    pred = gate.signed(out/"predictions.json", "id")
    release = gate.signed(root/"dataset-v1/dataset.json", "dataset_id")
    require(pred["plan_id"] == pid == release["plan_id"] and pred["model_digest"] == digest(read(args.model))
            and pred["blind_seal"] == digest(read(root/"validation/blind-seal.json")), "model or archive drift")
    if pred["prospective"]:
        require(read(root/"model-before-pairing.json")["model_digest"] == pred["model_digest"]
                == release["model_before_pairing"], "frozen MLP does not match prospectively forked run")
    _, rows = fp.checked_run(plan, "validation", root/"validation")
    samples, evidence, ledger = (gate.lines(root/"dataset-v1"/("validation"+s+".jsonl"))
                                 for s in ("", "-evidence", "-episodes"))
    info = release["splits"]["validation"]
    require([digest(x) for x in (samples,evidence,ledger)] ==
            [info[k+"_digest"] for k in ("samples","evidence","ledger")], "release hash mismatch")
    pairs = {e["sample_id"]:e for e in evidence}
    labels = {s["sample_id"]:s for s in samples}
    shadows = read(root/"audit-validation/shadow.json")
    require(shadows["plan_id"] == pid, "wrong retrieval accounting")
    totals = {k:Counter() for k in POLICIES}
    errors = defaultdict(list)
    entries = []
    for r, d, l, sh in zip(rows, pred["rows"], ledger, shadows["rows"], strict=True):
        require(r["episode_id"] == d["episode_id"] == l["episode_id"] == sh["episode_id"] and
                d["match"] == l["decision"] == sh["status"], "invalid row alignment")
        sid = l["sample_id"]
        bare = {"success":r["verifier"]["reward"],"input_tokens":r["input_tokens"],"output_tokens":r["output_tokens"],
                "total_tokens":r["input_tokens"]+r["output_tokens"],"tool_calls":r["tool_calls"],
                "actor_calls":sh["bare_actor_calls"],"accounted_seconds":sh["bare_actor_seconds"]}
        arms = {}
        if sid:
            sm, ev = labels[sid], pairs[sid]
            require(d["features_digest"] == digest(sm["features"]) and
                    ev["task_digest"] == digest(r["task"]) and ev["checkpoint"] == "validation/"+d["checkpoint"],
                    "decision features differ from paired checkpoint")
            for name, ref in ev["arms"].items():
                live = read(root/ref["path"])
                observed = ref["observed"]
                require(digest(live) == ref["digest"] and live["arm"] == name and
                        verify(r["task"], live["state"])["reward"] == live["score"]["reward"] == observed["success"] and
                        all(observed[k] == v for k,v in live["usage"].items()), "raw arm score or cost mismatch")
                arms[name] = observed
            measured = arms["remind"]["utility"] - arms["silent"]["utility"]
            require(math.isclose(measured, sm["targets"]["D"], abs_tol=1e-10), "target drift")
        elif d["predictions"] and "insufficient_revision_budget" not in l["exclusion"]:
            raise ValueError("An eligible checkpoint lacks a paired outcome; cannot score as silent")
        elif l["decision"] not in {"never_reached", "inapplicable", "no_candidate"}:
            require("insufficient_revision_budget" in l["exclusion"], "unknown outcome cannot be imputed")
        entry = {"episode_id":r["episode_id"],"sample_id":sid,"cell":pairs[sid]["cell"] if sid else None,
                 "policy":{}}
        for name in POLICIES:
            action = bool(sid and d["predictions"][name]["remind"])
            raw = arms["remind" if action else "silent"] if arms else bare
            cpu = d["gate_seconds"].get(name,0)
            metrics = score(raw, cpu + sh["match_seconds"], plan)
            require(metrics["utility"] is not None, "unobserved task or unknown priced cost")
            t = totals[name]
            t.update({k:metrics[k] for k in ("success","total_tokens","accounted_seconds","utility","tool_calls")})
            t.update(reminders=int(action), gate_seconds=cpu, episodes=1,
                     rescues=int(action and entry["cell"]=="rescue"), regressions=int(action and entry["cell"]=="regression"))
            entry["policy"][name] = {"remind":action, **metrics}
            if sid and "predicted_D" in d["predictions"][name]:
                errors[name].append(d["predictions"][name]["predicted_D"] - measured)
        entries.append(entry)
    require(len(samples) == len(pairs) == sum(bool(e["sample_id"]) for e in entries), "incomplete coverage")
    report = {"plan_id":pid,"model_digest":pred["model_digest"],"prospective":pred["prospective"],
              "episodes":len(entries),"pairs":len(samples),"cells":dict(Counter(e["cell"] for e in evidence)),
              "totals":totals, "conditional_D_errors":{k:{"mae":sum(abs(n) for n in vs)/len(vs),
              "rmse":math.sqrt(sum(n*n for n in vs)/len(vs))} for k,vs in errors.items()},
              "load_seconds":pred["load_seconds"],"ledger_digest":digest(entries)}
    formal.write_rows(out/"policy-episodes.jsonl",entries)
    save(out/"comparison.json",report)
    print(out/"comparison.json")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("mode", choices=("predict","score"))
    cli.add_argument("--root",type=Path,required=True)
    cli.add_argument("--model",type=Path,required=True)
    cli.add_argument("--stump",type=Path,default=fp.OUTPUT/"cave-gate-01/frozen/model.json")
    cli.add_argument("--out",type=Path,required=True)
    opt = cli.parse_args()
    (predict if opt.mode=="predict" else evaluate)(opt)
