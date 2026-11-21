"""Register an online, randomized, one-observed-arm mini cohort."""

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path

from . import formal_plan as fp, gate_validation as gv
from .records import digest, read, save
from .supplement_plan import SHAPES, make_input


def register(root, parent):
    prior, parent_id = fp.load(parent)
    sealed = read(parent / "sealed-test.json")
    seen = {tuple(sorted(t["task"]["input"])) for t in sealed["tasks"]}
    for path in sorted(parent.parent.glob("cave-*/**/manifest.json")):
        manifest = read(path)
        if manifest.get("benchmark") == "mini-data-v1":
            seen.update(tuple(sorted(t["input"])) for t in manifest["tasks"])
    historical = digest(sorted(seen))
    rows = []
    for i, (name, (family, text)) in enumerate(gv.TEMPLATES.items()):
        seed = 1010000 + 10000 * i
        for n in range(24):
            shape = SHAPES[n % len(SHAPES)]
            while True:
                seed += 1
                values = make_input(seed, shape, family)
                if values is not None and not any(fp.near_duplicate(values, t) for t in seen):
                    break
            seen.add(tuple(sorted(values)))
            task = {"template_id": name, "family": family, "instruction": text, "input": values}
            rows.append({
                "episode_id": f"ep-{len(rows):04d}", "seed": seed, "task": task,
                "task_digest": digest(task), "input_digest": digest(values),
                "multiset_digest": digest(sorted(values)), "input_stratum": shape, "q": .25,
                "main_opportunity": random.Random("checkpoint:" + str(seed)).randint(1, 4),
            })
    plan = {**prior, "schema": "randomized-single-arm/v1", "parent_plan_id": parent_id,
            "created_utc": datetime.now(timezone.utc).isoformat(), "past_inputs_digest": historical,
            "online_templates": gv.TEMPLATES, "registration_code": fp.code_digest("single_arm_plan.py"),
            "assignment_probabilities": {"silent": .5, "remind": .5},
            "split_counts": {"single_arm": len(rows), "test": 0},
            "cohorts": [{"split": "single_arm", "tasks": rows}],
            "design": "96 fresh episodes. Draw j uniformly before the episode, save the no-memory draft, then "
                      "use the frozen two-card router. At the selected eligible and funded decision record "
                      "features and real assignment propensities before executing one assigned action. "
                      "HMAC coin with batch-level key commitment and end-of-run reveal. Never clone, replay "
                      "or fork a target. No replacement, no pair labels, no gate-dependent assignment.",
            "outcome_contract": {"unit": "one registered task and at most one randomized decision",
                "population": "four existing mini task wordings by four input shapes, equal counts",
                "primary_log": "only actual eligible randomized draws; report eligibility funnel separately",
                "missing": "Keep assignment and reason if terminal reward or a priced cost is missing.",
                "cost": "Y minus priced actor tokens, tool calls and accounted online seconds; retain raw components.",
                "use": "card 10 logged-policy input; separate from paired-fork data"},
            "expected_test_digest": prior["test_manifest_digest"]}
    # Parent plans may carry earlier validation-run metadata. Only shared frozen settings apply here.
    for key in ("trial_registration", "prior_validation_plan_id", "validation_templates", "validation_code"):
        plan.pop(key, None)
    root.mkdir(parents=True, exist_ok=False)
    save(root / "sealed-test.json", sealed)
    save(root / "preregistration.json", {**plan, "plan_id": digest(plan)})
    print(digest(plan), len(rows))


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--parent", type=Path, default=fp.OUTPUT / "cave-mlp-validation-01")
    args = cli.parse_args()
    register(args.root, args.parent)
