"""Recompute live trajectory accounting and the dev- or test-suite denominator."""
import argparse
import math
from collections import Counter
from pathlib import Path

from .gate import signed
from .records import digest, read, save
from .replay import require


def audit(root):
    reg = signed(root/"registration.json","study_id")
    groups = read(root/"results.json")
    required = {f"actor-{i}/{s}" for i in range(len(reg["actor_weight_ids"])) for s in reg["systems"]}
    require(set(groups)==required,"incomplete system or actor coverage")
    summaries = {}
    for key, rows in groups.items():
        require([r["task_id"] for r in rows]==reg["task_ids"],"incomplete task denominator")
        entries = {v["task_digest"]:v for v in (read(p) for p in (root/key).glob("*/result.json"))}
        summary=Counter()
        mechanisms=Counter()
        for r in rows:
            ep=entries[r["task_id"]]
            dest=root/key/ep["episode_id"]
            require(ep["plan_id"]==reg["study_id"] and r["system"]==key.split("/")[1]
                    and r["arm"]==ep["assigned_arm"],"episode lineage")
            generations=[read(p) for p in dest.glob("*/*-generation.json")]
            usage=ep["usage"]
            if usage["usage_complete"]:
                require(len(generations)==usage["actor_calls"]<=reg["max_steps"]
                        and sum(g["prompt_tokens"]+g["completion_tokens"] for g in generations)==usage["total_tokens"]==r["tokens"]
                        and sum(len((read(p)["executed_action"] or {}).get("commands",[])) for p in dest.glob("*/execution.json"))
                        ==usage["tool_calls"]<=reg["total_tools"],"receipt mismatch")
            require(ep["verifier"]==read(dest/"verdict.json") and ep["verifier"]["reward"]==r["reward"],"verifier mismatch")
            if r["utility"] is not None:
                require(math.isclose(r["utility"],r["reward"]-usage["cost_penalty"]), "utility mismatch")
            summary.update(tasks=1,scored=int(r["reward"] is not None),success=int(r["reward"]==1),
                           reminders=int(r["arm"]=="remind"),available_tokens=r["tokens"] or 0,
                           missing_cost=int(r["tokens"] is None))
            mechanisms[r["mechanism"]]+=1
        summaries[key]={**summary,"by_mechanism":mechanisms}
    report={"source":digest(groups),"study_id":reg["study_id"],"split":reg["split"],"systems":summaries}
    save(root/"audit.json",report)
    return report


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--root",required=True,type=Path)
    args=cli.parse_args()
    print(audit(args.root))
