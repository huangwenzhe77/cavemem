"""Register a training-only acquisition batch against the unchanged formal study."""

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path

from . import formal_plan as fp
from .records import digest, read, save
from .replay import require


GROUP_COUNTS = {"ledger-down": 80, "archive-down": 48, "ledger-up": 16, "archive-up": 16}
SHAPES = ("ordinary", "positive", "already_ordered", "one_swap")


def make_input(seed, shape, family):
    rng = random.Random(seed)
    lower = 1 if shape == "positive" else -9
    values = [rng.randint(lower, 22) for _ in range(8)]
    values[-1] = values[0]
    if shape in {"already_ordered", "one_swap"}:
        values.sort(reverse=family == "descending")
        if shape == "one_swap":
            unequal = [(i, j) for i in range(len(values)) for j in range(i) if values[i] != values[j]]
            if not unequal:
                return None
            i, j = rng.choice(unequal)
            values[i], values[j] = values[j], values[i]
    return values


def freeze(root, parent):
    previous, parent_id = fp.load(parent)
    release = read(parent / "dataset-v1/dataset.json")
    require(release["plan_id"] == parent_id and release["dataset_id"] ==
            digest({k: v for k, v in release.items() if k != "dataset_id"}), "parent dataset mismatch")
    sealed = read(parent / "sealed-test.json")
    seen = [r["task"]["input"] for c in previous["cohorts"] for r in c["tasks"]]
    seen.extend(r["task"]["input"] for r in sealed["tasks"])
    for path in sorted(parent.parent.glob("cave-*/**/manifest.json")):
        manifest = read(path)
        if manifest.get("benchmark") == "mini-data-v1":
            seen.extend(t["input"] for t in manifest["tasks"])
    seen = [list(key) for key in sorted({tuple(sorted(t)) for t in seen})]
    excluded_digest = digest(seen)
    rows = []
    groups = {g[1]: g for g in fp.GROUPS if g[0] == "train"}
    for idx, (name, count) in enumerate(GROUP_COUNTS.items()):
        _, _, family, _, _, instruction = groups[name]
        seed = 410000 + idx * 10000
        for n in range(count):
            shape = SHAPES[n % len(SHAPES)]
            while True:
                seed += 1
                values = make_input(seed, shape, family)
                if values is not None and not any(fp.near_duplicate(values, t) for t in seen):
                    break
            seen.append(values)
            task = {"template_id": name, "family": family, "instruction": instruction, "input": values}
            rows.append({
                "episode_id": f"ep-{len(rows):04d}", "task": task, "seed": seed,
                "task_digest": digest(task), "input_digest": digest(values),
                "multiset_digest": digest(sorted(values)), "input_stratum": shape, "q": .25,
                "main_opportunity": random.Random("checkpoint:" + str(seed)).randint(1, 4),
            })
    plan = {
        **previous, "schema": "formal-training-supplement/v1", "parent_plan_id": parent_id,
        "parent_dataset_id": release["dataset_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "registration_code": fp.code_digest("supplement_plan.py"), "prior_input_multisets_digest": excluded_digest,
        "cohorts": [{"split": "train", "tasks": rows}], "split_counts": {"train": len(rows), "validation": 0, "test": 0},
        "design": "Fixed training enrichment. The same four training wordings and uniform opportunity law; "
                  "new inputs stratified by ordering and sign, oversampling the ledger-down wording. "
                  "All tasks included; no targeting by actual checkpoint action, verdict or reminder response. "
                  "No interim stopping, re-draw or outcome-based top-up. No new validation or test episodes.",
        "target_rule_version": "more-expensive-silent-pass/v1",
        "target_definition": "Both targets require silent success and strictly positive reminder-minus-silent "
                             "total token, accounted seconds and priced cost deltas. Split on reminder failure "
                             "versus reminder success. Retain every complete pair and original D/H.",
    }
    root.mkdir(parents=True, exist_ok=False)
    save(root / "sealed-test.json", sealed)
    save(root / "preregistration.json", {**plan, "plan_id": digest(plan)})
    print("registered", len(rows), digest(plan), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--parent", type=Path, default=Path("outputs/cave-formal-01"))
    args = cli.parse_args()
    freeze(args.root, args.parent)
