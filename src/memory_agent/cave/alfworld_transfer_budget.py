"""Select a shared service budget on independent training games, then score frozen official traces."""

import argparse
from math import ceil
from pathlib import Path
from types import SimpleNamespace

from .alfworld_env import sha_file
from .alfworld_study_rollout import receipt
from .alfworld_transfer_audit import failure_interval, harmed_cases, redraft_contrast
from .alfworld_transfer_probe import load_registry
from .alfworld_transfer_study import audit_pair, score_tokens, summary
from .records import read, save


PERCENTILES = (35, 45, 55, 65, 75, 85)
EXPECTED_OFFICIAL_COUNTS = {"valid_seen": 140, "valid_unseen": 134}


def read_study(study):
    frozen = read(study / "transfer-freeze.json")
    base = Path(frozen["baseline"])
    artifact_hashes = {
        base / "manifest.json": frozen["manifest_sha256"],
        study / "transfer-bank.json": frozen["bank_sha256"],
        **{Path(__file__).with_name(n): h for n, h in frozen["code"].items()},
    }
    if any(sha_file(path) != digest for path, digest in artifact_hashes.items()):
        raise ValueError("the underlying study has changed")
    manifest = read(base / "manifest.json")
    cards = {row["id"] for row in read(study / "transfer-bank.json")["cards"]}
    return frozen, base, manifest, cards


def score_records(records, budget):
    result = []
    for spec, natural, pair, branches in records:
        row = score_tokens(natural, pair["match"], branches, budget)
        if any(row[arm]["tokens"] > budget for arm in ("silent", "neutral", "episodic")):
            raise ValueError("a scored arm exceeded the shared token limit")
        row["spec"] = spec["id"]
        result.append((spec["template"], row))
    return result


def read_pairs(specs, natural_root, paired_root, cards):
    result = []
    for spec in specs:
        natural = read(receipt(natural_root, spec))
        if natural["spec"] != spec or natural["extra_draft"] is not None:
            raise ValueError("recorded silent arm differs from the frozen registry")
        pair = read(receipt(paired_root, spec, "paired"))
        verified = audit_pair(SimpleNamespace(out=paired_root), spec, natural, pair, cards)
        result.append((spec, natural, pair, verified))
    return result


def freeze_budget(args, manifest, cards):
    if args.probe is None:
        raise ValueError("selecting a budget requires independent training probe traces")
    probe_lock = read(args.probe / "probe-freeze.json")
    study_hash = sha_file(args.study / "transfer-freeze.json")
    if (probe_lock["study_sha256"] != study_hash
            or probe_lock["cohort_sha256"] != sha_file(args.probe / "manifest.json")):
        raise ValueError("probe evidence was generated under a different policy or registry")
    specs = load_registry(SimpleNamespace(out=args.probe), manifest)
    if read(args.probe / "probe-audited.json")["tasks"] != len(specs):
        raise ValueError("incomplete audited training probe")
    records = read_pairs(specs, args.probe, args.probe, cards)
    wins = sorted(n["metrics"]["tokens"] for _, n, _, _ in records if n["metrics"]["success"])
    if not wins:
        raise ValueError("no silent successes for budget calibration")
    options = []
    for percentile in PERCENTILES:
        index = ceil(percentile * len(wins) / 100) - 1
        budget = wins[index]
        rows = score_records(records, budget)
        counts = summary([row for _, row in rows])
        options.append({"percentile": percentile, "token_limit": budget, "counts": counts,
                        "conditional": failure_interval(rows, manifest["seed"])})
    viable = [r for r in options if r["counts"]["silent_wins"] >= 5
              and .4 <= r["conditional"]["rate"] <= .6]
    winner = max(viable, key=lambda r: (
        r["counts"]["harmed_by_content"], r["counts"]["silent_wins"],
        -abs(r["conditional"]["rate"] - .5),
    ), default=None)
    protocol = {
        "source": "template-disjoint training games; no official results read during budget selection",
        "study_sha256": study_hash, "probe_sha256": sha_file(args.probe / "probe-audited.json"),
        "training_options": options, "selected": winner,
        "rule": "40–60% with at least five silent successes, then content-specific harms, denominator and proximity to 50%",
    }
    path = args.out / "budget-freeze.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("budget cannot change after freezing")
    save(path, protocol)
    print("SELECTED", winner, flush=True)


def score_official(args, frozen, base, manifest, cards):
    selected = read(args.out / "budget-freeze.json")
    if selected["study_sha256"] != sha_file(args.study / "transfer-freeze.json"):
        raise ValueError("budget study was frozen against another rollout policy")
    if selected["selected"] is None:
        raise RuntimeError("no eligible resource budget was selected on training")
    established = args.study / "audited-summary.json"
    if read(established)["tasks"] != 274:
        raise ValueError("a complete prior official trajectory audit is required")
    budget = selected["selected"]["token_limit"]
    result = {}
    overall = []
    full_by_id = {}
    for group, expected in EXPECTED_OFFICIAL_COUNTS.items():
        specs = manifest["splits"][group]
        if len(specs) != expected or len({s["id"] for s in specs}) != expected:
            raise ValueError("the official manifest differs from the complete cohort")
        records = read_pairs(specs, base, args.study, cards)
        full_by_id.update({spec["id"]: pair for spec, _, pair, _ in records})
        rows = score_records(records, budget)
        overall.extend(rows)
        result[group] = {"counts": summary([row for _, row in rows]),
                         "conditional": failure_interval(rows, manifest["seed"]),
                         "redraft_contrast": redraft_contrast(rows)}
    result["overall"] = {"counts": summary([row for _, row in overall]),
                         "conditional": failure_interval(overall, manifest["seed"]),
                         "redraft_contrast": redraft_contrast(overall)}
    result.update({"cases": harmed_cases(overall, full_by_id), "tasks": len(overall),
                   "token_budget": budget, "prior_official_audit_sha256": sha_file(established),
                   "previous_official_exposure": True, "same_paired_trajectories_rescored": True,
                   "arms": frozen["arms"]})
    save(args.out / "budget-audited.json", result)
    print("TRANSFER BUDGET AUDITED", len(overall), budget, result["overall"]["conditional"])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("freeze", "score"))
    cli.add_argument("--study", required=True, type=Path)
    cli.add_argument("--out", required=True, type=Path)
    cli.add_argument("--probe", type=Path)
    args = cli.parse_args()
    study, base, manifest, cards = read_study(args.study)
    if args.stage == "freeze":
        freeze_budget(args, manifest, cards)
    else:
        score_official(args, study, base, manifest, cards)
