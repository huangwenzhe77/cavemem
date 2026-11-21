"""Register fresh validation instances for the MLP before running either arm."""

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path

from . import formal_plan as fp, gate_validation as gv
from .records import digest, read, save
from .supplement_plan import SHAPES, make_input


def register(root, parent):
    prior, prior_id = fp.load(parent)
    sealed = read(parent / "sealed-test.json")
    seen = {tuple(sorted(t["task"]["input"])) for t in sealed["tasks"]}
    for manifest in sorted(parent.parent.glob("cave-*/**/manifest.json")):
        run = read(manifest)
        if run.get("benchmark") == "mini-data-v1":
            seen.update(tuple(sorted(t["input"])) for t in run["tasks"])
    history = digest(sorted(seen))
    rows = []
    for i, (name, (family, instruction)) in enumerate(gv.TEMPLATES.items()):
        seed = 810000 + 10000 * i
        for n in range(16):
            shape = SHAPES[n % len(SHAPES)]
            while True:
                seed += 1
                values = make_input(seed, shape, family)
                if values is not None and not any(fp.near_duplicate(values, v) for v in seen):
                    break
            seen.add(tuple(sorted(values)))
            task = {"template_id": name, "family": family, "instruction": instruction, "input": values}
            rows.append({
                "episode_id": f"ep-{len(rows):04d}", "seed": seed, "task": task,
                "task_digest": digest(task), "input_digest": digest(values),
                "multiset_digest": digest(sorted(values)), "input_stratum": shape, "q": .25,
                "main_opportunity": random.Random("checkpoint:" + str(seed)).randint(1, 4),
            })
    plan = {**prior, "cohorts": [{"split": "validation", "tasks": rows}],
            "created_utc": datetime.now(timezone.utc).isoformat(), "past_inputs_digest": history,
            "prior_validation_plan_id": prior_id, "trial_registration": fp.code_digest("mlp_trial.py"),
            "design": "Fresh inputs under four existing validation templates and four registered input strata. "
                      "All 64 episodes, one uniform opportunity, no refill. Train on released training rows only. "
                      "Freeze MLP, ablation, rule, tree stump and all treatment decisions before branching. "
                      "Use threshold 0.002; report success, tokens, utility, arm costs plus gate overhead."}
    root.mkdir(parents=True, exist_ok=False)
    save(root / "sealed-test.json", sealed)
    save(root / "preregistration.json", {**plan, "plan_id": digest(plan)})
    print("MLP validation plan", digest(plan), len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=fp.OUTPUT / "cave-gate-validation-01")
    args = parser.parse_args()
    register(args.root, args.parent)
