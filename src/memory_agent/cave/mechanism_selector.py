"""Pre-outcome, prefix-only sampling rules for a separate training cohort."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import formal_plan as fp
from .records import digest, read, save
from .replay import require, restore


def terminal_or_budget_boundary(task, pre, draft, plan):
    """Recognize two ordinary DSL patterns without running the draft or a verifier."""
    t = pre["step"]
    if t >= plan["max_steps"] or draft["error"]:
        return None
    state = pre["state"]
    if state["submitted"] is not None or Counter(state["working"]) != Counter(task["input"]):
        return None
    desc = task["family"] == "descending"
    sort_op = "DESC" if desc else "ASC"
    ops = [c["keystrokes"].strip() for c in draft["a0"]["commands"]]
    if not ops or not set(ops) <= {sort_op, "SHOW", "SUBMIT"}:
        return None
    ordered = state["working"] == sorted(state["working"], reverse=desc)
    if ops.count("SUBMIT") == 1 and ops[-1] == "SUBMIT" and (ordered or sort_op in ops[:-1]):
        return "ready_to_submit"
    if ("SUBMIT" not in ops and plan["max_steps"] - t == 1 and not draft["a0"]["complete"]
            and (sort_op in ops or ordered)):
        return "one_call_left_to_submit"
    return None


def choose(run, rows, plan, cards):
    choices = []
    for row in rows:
        result = {"episode_id": row["episode_id"], "status": "no_qualifying_prefix"}
        for node in row["steps"]:
            if not node["executable"]:
                continue
            pre, draft, _ = restore(run, row, node["turn_id"])
            kind = terminal_or_budget_boundary(row["task"], pre, draft, plan)
            if kind is None:
                continue
            decision = fp.formal_route(cards, row["task"], pre, draft)
            if decision.get("rho") != 1:
                continue
            result.update(decision, turn=pre["step"], pre_digest=digest(pre), a0_digest=draft["a0_digest"],
                          checkpoint_id=pre["checkpoint_id"], mechanism=kind)
            break
        choices.append(result)
    return choices


def register(root):
    plan, pid = fp.load(root)
    require(not (root/"audit-train").exists(), "register before any paired runs")
    out = root/"mechanism"
    out.mkdir(exist_ok=False)
    settings = {
        "version": "first-applicable-ready-or-boundary/v1", "plan_id": pid,
        "registered_utc": datetime.now(timezone.utc).isoformat(), "selector_sha": fp.code_digest("mechanism_selector.py"),
        "size": len(fp.tasks_for(plan,"train")), "policy": "first qualifying observable prefix in order; one per episode",
        "eligibility": "Applicable frozen card, intact data, exact typed sorting/SHOW/terminal SUBMIT pattern. "
                       "Either the cached draft can submit the full ordered input now or at the penultimate model "
                       "call it preserves a sorted result for a remaining submit. No terminal verifier or treatment "
                       "response enters selection. Do not change any caller budgets or reminders.",
        "overlap": "Reuse the primary pair when both designs select the same node; no second measurement, no refill.",
        "targets": plan["target_definition"],
        "interpretation": "Mechanism-enriched training cohort. No natural-population rate; cluster by original task.",
    }
    save(out/"preregistration.json",{**settings,"study_id":digest(settings)})


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--root",type=Path,required=True)
    register(cli.parse_args().root)
