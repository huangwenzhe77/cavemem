"""Lock and evaluate a short-reminder gate that only knows memory properties."""

import argparse
from pathlib import Path

import torch

from memory_agent.cave.alfworld_env import sha_file
from memory_agent.cave.alfworld_gate_refit import policy_report
from memory_agent.cave.alfworld_gate_test import interval
from memory_agent.cave.alfworld_stage_policy import decide, fit, predict, select
from memory_agent.cave.records import read, save


MEMORY_AND_BUDGET = {14, 15, 16, 17, 19, 20, 21, 22}
SETTING = "p50"


def remove_state(rows):
    return [{**r, "features": [v if i in MEMORY_AND_BUDGET else 0 for i, v in enumerate(r["features"])]}
            if r["features"] is not None else r for r in rows]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("lock", "score"))
    for name in ("out", "development", "previous"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args()
    model_path = args.development / "model.json"
    lock = args.out / "memory-control-freeze.json"
    seal = args.out / "sealed-test.json"
    report = args.out / "test-report.json"
    main_model = read(model_path)
    hashes = {"model": sha_file(model_path), "seal": sha_file(seal), "code": sha_file(Path(__file__))}
    if args.phase == "lock":
        if lock.exists() or report.exists():
            raise ValueError("memory control must be locked before opening the extension")
        torch.set_num_threads(1)
        labels = args.development / "paired-labels.json"
        if sha_file(labels) != main_model["dataset_sha256"]:
            raise ValueError("short reminder training data has changed")
        data = read(labels)
        spec = main_model["model"]
        train = remove_state([r for r in data["train"] if r["features"] is not None])
        val = remove_state(data["validation"])
        comparator = fit(train, spec["width"], spec["penalty"])
        rules = select(val, predict(val, comparator))
        save(lock, {"hashes": hashes, "model": comparator, "rules": rules})
        print("SHORT_MEMORY_CONTROL_LOCKED", rules[SETTING], flush=True)
        return
    frozen = read(lock)
    if frozen["hashes"] != hashes or not report.exists():
        raise ValueError("extension is incomplete or its comparator changed")
    older = [r for p in args.previous.glob("stage-test/*/scored.json") for r in read(p)["rows"] if r["setting"] == SETTING]
    newer = [read(p) for p in args.out.glob("place-test/*/scored.json")]
    if len(older) != 167 or len(newer) != len(read(seal)["splits"]["test"]):
        raise ValueError("partial or mismatched short test")
    result = {}
    for name, rows in (("independent_extension", newer), ("pooled", older + newer)):
        control_rows = remove_state(rows)
        guide = decide(control_rows, predict(control_rows, frozen["model"]), frozen["rules"])
        full = decide(rows, predict(rows, main_model["model"]), main_model["rules"])
        diff = [int(a) - int(b) for a, b in zip(full, guide, strict=True)]
        result[name] = {"memory_only": policy_report(rows, guide)[SETTING],
                        "full": policy_report(rows, full)[SETTING],
                        "success_gain": interval(rows, [d * (r["R"] - r["H"]) for r, d in zip(rows, diff, strict=True)]),
                        "token_saving": interval(rows, [-d * r["extra"] for r, d in zip(rows, diff, strict=True)])}
    save(args.out / "memory-control-report.json", result)
    print("SHORT_MEMORY_CONTROL_COMPLETE", result["pooled"]["success_gain"], flush=True)


if __name__ == "__main__":
    main()
