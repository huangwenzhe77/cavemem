"""Replay a frozen reminder policy on independent games from the ALFWorld training split."""

import argparse
from pathlib import Path

from .alfworld_env import Actor, sha_file
from .alfworld_ranker import RankedActor, examples, load
from .alfworld_study_pairs import cached
from .alfworld_study_rollout import receipt, rollout
from .alfworld_transfer_audit import failure_interval, redraft_contrast
from .alfworld_transfer_study import audit_pair, compare, score_tokens, summary
from .records import read, save


def load_registry(args, study_manifest):
    probe_manifest = read(args.out / "manifest.json")
    if probe_manifest["config_sha256"] != study_manifest["config_sha256"]:
        raise ValueError("probe and frozen environment configurations differ")
    specs = probe_manifest["splits"]["development"]
    used = [s for rows in study_manifest["splits"].values() for s in rows]
    for key in ("template", "game", "sha256"):
        proposed = {s[key] for s in specs}
        if len(proposed) != len(specs) or proposed.intersection(s[key] for s in used):
            raise ValueError(f"probe cohort is not independent by {key}")
    if len({s["id"] for s in specs}) != len(specs):
        raise ValueError("duplicate probe game IDs")
    if not specs or any(not s["game"].startswith("json_2.1.1/train/") for s in specs):
        raise ValueError("probe games must belong to the official training split")
    return specs


def probe(args):
    study_file = args.study / "transfer-freeze.json"
    frozen = read(study_file)
    base = Path(frozen["baseline"])
    study_manifest = read(base / "manifest.json")
    specs = load_registry(args, study_manifest)
    if args.out.resolve() in (args.study.resolve(), base.resolve()):
        raise ValueError("provide a separate probe directory")
    config = read(base / "actor.json")
    bank_file = args.study / "transfer-bank.json"
    versioned = [
        (args.config, study_manifest["config_sha256"]),
        (base / "manifest.json", frozen["manifest_sha256"]),
        (base / "actor.json", frozen["actor_sha256"]),
        (base / "ranker.pt", config["ranker"]),
        (bank_file, frozen["bank_sha256"]),
        *[(Path(__file__).with_name(name), checksum) for name, checksum in frozen["code"].items()],
    ]
    if str(args.weights.resolve()) != config["actor"] or any(sha_file(p) != c for p, c in versioned):
        raise ValueError("probe runtime differs from the frozen study")
    lock = args.out / "probe-freeze.json"
    run_spec = {"study": str(args.study.resolve()), "study_sha256": sha_file(study_file),
                "cohort_sha256": sha_file(args.out / "manifest.json"), "sets_new_thresholds": False}
    if lock.exists() and read(lock) != run_spec:
        raise ValueError("probe inputs changed after freezing")
    save(lock, run_spec)
    args.exposure = frozen["exposure"]
    args.representation = frozen.get("representation", "trajectory")
    ranker = load(base)
    actor = RankedActor(Actor(args.weights), ranker, config["top"], config.get("repeat_aware", False),
                        examples(base, ranker) if config.get("examples") else (), config["indexed"])
    cards = read(bank_file)["cards"]
    ids = {card["id"] for card in cards}
    budgets = frozen["token_budgets"]
    observed = {k: [] for k in ("calls", *budgets)}
    for spec in specs:
        natural = cached(receipt(args.out, spec), lambda: rollout(args, spec, actor))
        if natural["spec"] != spec or natural["extra_draft"] is not None:
            raise ValueError("probe baseline does not match the manifest")
        pair = cached(receipt(args.out, spec, "paired"), lambda: compare(
            args, actor, spec, natural, cards, study_manifest["seed"],
        ))
        branches = audit_pair(args, spec, natural, pair, ids)
        observed["calls"].append((spec["template"], pair))
        for setting, budget in budgets.items():
            selected = score_tokens(natural, pair["match"], branches, budget)
            observed[setting].append((spec["template"], selected))
        print("TRANSFER_PROBE", spec["id"],
              [pair[arm]["success"] for arm in frozen["arms"]], flush=True)
        save(args.out / "probe-progress.json", {
            setting: summary([row for _, row in values]) for setting, values in observed.items()
        })
    result = {"tasks": len(specs), "study": run_spec, "budgets": budgets}
    for setting, values in observed.items():
        result[setting] = {
            "counts": summary([row for _, row in values]),
            "failure_rate": failure_interval(values, study_manifest["seed"]),
            "redraft_contrast": redraft_contrast(values),
        }
    save(args.out / "probe-audited.json", result)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for label in ("study", "out", "data", "config", "weights"):
        cli.add_argument(f"--{label}", required=True, type=Path)
    probe(cli.parse_args())
