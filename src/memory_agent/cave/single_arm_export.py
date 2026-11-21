"""Verify a sealed forward-only run and export observed decisions for policy evaluation."""

import argparse
import hashlib
import hmac
import math
import tempfile
from collections import Counter
from pathlib import Path

from . import formal, formal_plan as fp
from .gate import lines
from .logged_data import write_dataset
from .records import digest, read
from .replay import require


def verify_draw(secret, pid, row, selection, assignment, rid):
    msg = b"CAVE-online-single-arm/v1\0" + pid.encode() + b"\0" + row["episode_id"].encode()
    mac = hmac.digest(secret, msg, "sha256")
    n = int.from_bytes(mac[:8], "big")
    arm = "remind" if n < 2**63 else "silent"
    require(assignment == {
        "arm": arm, "Z": int(arm == "remind"), "propensities": {"remind": .5, "silent": .5},
        "observed_propensity": .5, "uniform_numerator": str(n), "uniform_denominator": str(2**64),
        "hmac_sha256": mac.hex(), "episode_id": row["episode_id"], "plan_id": pid, "run_id": rid,
        "checkpoint_id": selection["checkpoint_id"], "selection_digest": digest(selection),
        "drawn_before_execution": True,
    }, "assignment does not open against the committed key and saved selection")


def audit_episode(folder, row, summary, plan, pid, rid, secret):
    raw = {p.relative_to(folder).as_posix(): read(p) for p in folder.rglob("*.json")
           if p != folder / "summary.json"}
    require(read(folder / "summary.json") == summary and
            {k: digest(v) for k, v in raw.items()} == summary["event_digests"], "event ledger differs")
    eid = row["episode_id"]
    require(summary["episode_id"] == eid and summary["task_digest"] == digest(row["task"])
            and summary["run_id"] == rid and summary["plan_id"] == pid
            and summary["seed"] == row["seed"] and summary["q"] == row["q"] == .25
            and summary["main_opportunity"] == row["main_opportunity"], "episode lineage")
    usage = summary["usage"]
    interrupted = summary["termination"] == "interrupted"
    y = summary["verifier"]["reward"]
    seen = None
    draw = None
    state = None
    history = None
    tokens_in = tokens_out = calls = revisions = tools = opportunities = 0
    actor_seconds = tool_seconds = 0.
    for name in sorted(k for k in raw if k.endswith("/pre.json")):
        prefix = name.split("/")[0] + "/"
        pre = raw[name]
        require(pre["checkpoint_id"] == eid + "/" + prefix[:-1] and
                (state is None or pre["state"] == state) and
                (history is None or pre["messages"] == history), "broken factual trajectory")
        state = pre["state"]
        require((pre["used_input_tokens"], pre["used_output_tokens"], pre["used_tool_calls"])
                == (tokens_in, tokens_out, tools), "prefix budget mismatch")
        draft = raw.get(prefix + "draft.json")
        stage = "suffix" if draw else "draft"
        if draft:
            executable = bool(draft["a0"]["commands"]) and not draft["error"]
            opportunities += int(executable)
            require(draft["a0_digest"] == digest(draft["a0"]), "draft action changed")
            require(draft["messages"] == pre["messages"] + [
                {"role": "assistant", "content": raw[prefix + stage + "-generation.json"]["text"]}],
                "draft transcript changed")
        sel = raw.get(prefix + "selection.json")
        if draft and executable and opportunities == row["main_opportunity"] and not interrupted:
            require(sel or summary["termination"] == "infrastructure_error", "scheduled selection missing")
        if sel:
            require(not seen and draft and opportunities == row["main_opportunity"]
                    and sel["opportunity"] == opportunities and draft["a0"]["commands"] and not draft["error"]
                    and sel["pre_digest"] == digest(pre) and sel["draft_digest"] == digest(draft)
                    and sel["a0_digest"] == digest(draft["a0"]), "selection after wrong draft")
            seen = sel
            d = sel["decision"]
            expected = formal.gate_inputs(row, pre, draft, d, plan) if d.get("memory_id") else None
            require(sel["features"] == expected, "unapproved policy features")
            if d.get("rho") == 1:
                u = sel["usage_after_draft"]
                funded = (u["actor_calls"] < plan["max_steps"] and
                          u["input_tokens"] + u["output_tokens"] + sel["revision_prompt_tokens"] +
                          plan["actor"]["max_new_tokens"] <= plan["total_tokens"] and
                          u["tool_calls"] + len(draft["a0"]["commands"]) <= plan["total_tools"])
                require(sel["eligibility"] == ("eligible" if funded else "insufficient_revision_budget"),
                        "pre-assignment budget")
            draw = raw.get(prefix + "assignment.json")
            if draw:
                require(sel["eligibility"] == "eligible" and d["rho"] == 1, "ineligible assignment")
                verify_draw(secret, pid, row, sel, draw, rid)
            elif not interrupted:
                require(sel["eligibility"] != "eligible", "missing draw")
        for gen_stage in ("draft", "suffix", "revision"):
            req = raw.get(prefix + gen_stage + "-request.json")
            gen = raw.get(prefix + gen_stage + "-generation.json")
            if not req:
                require(not gen, "generation without request")
                continue
            calls += 1
            revisions += gen_stage == "revision"
            if gen_stage == "revision":
                require(sel and draw and draw["arm"] == "remind", "unassigned reminder")
                reminder = "\n\nAdvisory note for this proposal only: " + sel["decision"]["reminder"]
                expected = [*pre["messages"][:-1], {**pre["messages"][-1],
                            "content": pre["messages"][-1]["content"] + reminder}]
            else:
                require(gen_stage == stage, "unexpected generation")
                expected = pre["messages"]
            require(req["messages"] == expected, "actor prompt changed")
            if gen:
                require(gen["prompt_tokens"] == req["prompt_tokens"], "prompt account")
                tokens_in += gen["prompt_tokens"]
                tokens_out += gen["completion_tokens"]
                actor_seconds += gen["seconds"]
        ex = raw.get(prefix + "execution.json")
        if ex:
            chosen = raw[prefix + "chosen-action.json"]
            if sel and draw and draw["arm"] == "remind":
                stage = "revision"
            else:
                require(chosen["action"] == draft["a0"], "silent action differs from cached draft")
            require(chosen["stage"] == ex["stage"] == stage and ex["executed_action"] ==
                    (chosen["action"] if chosen["action"]["commands"] and not chosen["parse_error"] else None),
                    "executed action mismatch")
            tools += len((ex["executed_action"] or {}).get("commands", []))
            tool_seconds += ex["tool_seconds"]
            state = ex["state"]
            history = pre["messages"] + [
                {"role": "assistant", "content": raw[prefix + stage + "-generation.json"]["text"]},
                {"role": "user", "content": "\n".join(ex["observation"])}]
    require(summary["assignment"] == draw and sum(k.endswith("/assignment.json") for k in raw)
            == int(draw is not None) and revisions <= 1, "multiple or unmatched assignments")
    if y is not None:
        t = row["task"]
        answer = sorted(t["input"], reverse=t["family"] == "descending")
        require(raw["verdict.json"] == summary["verifier"] and y ==
                int(state is not None and state["submitted"] == answer), "terminal grade")
    if usage["usage_complete"]:
        require([calls, revisions, tools, tokens_in, tokens_out, tokens_in + tokens_out] ==
                [usage[k] for k in ("actor_calls", "revision_calls", "tool_calls", "input_tokens",
                                     "output_tokens", "total_tokens")], "full episode account")
        require(calls <= plan["max_steps"] and tools <= plan["total_tools"] and
                tokens_in + tokens_out <= plan["total_tokens"], "budget exceeded")
        require(math.isclose(actor_seconds, usage["actor_seconds"]) and
                math.isclose(tool_seconds, usage["tool_seconds"]), "time receipt")
        seconds = actor_seconds + tool_seconds + sum(usage[k] for k in
                    ("matching_seconds", "assignment_seconds", "prompt_count_seconds"))
        cost = (plan["lambda_tokens"] * (tokens_in + tokens_out) / 1000 + plan["lambda_tool"] * tools
                + plan["lambda_time"] * seconds / 60)
        require(math.isclose(seconds, usage["accounted_seconds"]) and math.isclose(cost, usage["cost_penalty"])
                and (y is None or math.isclose(y - cost, usage["observed_utility"])), "utility account")
    if not usage["usage_complete"] or y is None:
        require(usage["observed_utility"] is None, "unpriced or unscored utility")
    base = {"episode_id": eid, "cluster_id": row["task_digest"], "task_digest": row["task_digest"],
            "template_id": row["task"]["template_id"], "input_stratum": row["input_stratum"],
            "decision_id": digest([rid, draw["checkpoint_id"]]) if draw else None,
            "assigned_arm": draw["arm"] if draw else None, "q_position": row["q"],
            "scheduled_opportunity": row["main_opportunity"], "status": summary["main_status"],
            "termination": summary["termination"], "y": y, "r": usage["observed_utility"], "usage": usage,
            "verifier_status": summary["verifier"]["status"], "seed": row["seed"],
            "summary_digest": digest(summary)}
    logged = None
    if draw:
        logged = {**base, "z": draw["Z"], "arm": draw["arm"], "b": draw["propensities"],
                  "p_logged": draw["observed_propensity"], "x": seen["features"],
                  "features_digest": digest(seen["features"]), "checkpoint_id": draw["checkpoint_id"],
                  "selection_digest": digest(seen), "assignment_digest": digest(draw)}
    return base, logged


def export(root, dest):
    plan, pid = fp.load(root)
    run = root / "single_arm"
    mf, seal, reveal = (read(run / n) for n in ("manifest.json", "seal.json", "randomization-reveal.json"))
    index = lines(run / "index.jsonl")
    rows = plan["cohorts"][0]["tasks"]
    secret = bytes.fromhex(reveal["secret_hex"])
    require(plan["schema"] == "randomized-single-arm/v1" and len(secret) == 32
            and pid == mf["plan_id"] == reveal["plan_id"] and reveal["seal_digest"] == digest(seal)
            and hashlib.sha256(secret).hexdigest() == mf["commitment_sha256"] == reveal["commitment_sha256"]
            and mf["collector_code"] == fp.code_digest("single_arm.py")
            and mf["route_code"] == fp.code_digest("gate_validation.py")
            and mf["bank_id"] == plan["bank_id"] and mf["actor"] == plan["actor"]
            and mf["episode_ids"] == [r["episode_id"] for r in rows]
            and seal == {"plan_id": pid, "manifest_digest": digest(mf), "index_digest": digest(index),
                         "episodes": len(rows)}, "invalid run seal")
    episodes, logged = [], []
    for row, item in zip(rows, index, strict=True):
        ep, decision = audit_episode(run / row["episode_id"], row, item, plan, pid, mf["run_id"], secret)
        episodes.append(ep)
        if decision is not None:
            logged.append(decision)
    meta = {"plan_id": pid, "run_id": mf["run_id"], "seal_digest": digest(seal), "bank_id": mf["bank_id"],
            "status_counts": dict(Counter(e["status"] for e in episodes)),
            "arm_counts": dict(Counter(e["arm"] for e in logged)),
            "missing_assigned_utility": sum(e["r"] is None for e in logged),
            "environment": "mini-data-v1", "trajectory_mode": "forward-only"}
    require(not dest.exists(), "release already exists")
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        path = Path(tmp) / "release"
        path.mkdir()
        write_dataset(path, logged, episodes, meta)
        path.rename(dest)
    return read(dest / "dataset.json")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    print(export(args.root, args.out))
