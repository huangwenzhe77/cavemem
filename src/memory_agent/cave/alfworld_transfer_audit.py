"""Audit source evidence and independently reproduce the transfer study metrics."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .alfworld_env import sha_file
from .alfworld_study_rollout import receipt
from .alfworld_transfer_study import audit_pair, score_tokens, summary
from .records import read, save


def check_sources(root, source, bank, manifest):
    specs = {spec["id"]: spec for spec in manifest["splits"]["historical"]}
    for card in bank["cards"]:
        path = source / card["evidence"]
        if sha_file(path) != card["source_proof_sha256"]:
            raise ValueError("source evidence changed")
        proof = read(path)
        anchor = proof["anchor"]
        if len(proof["metrics"]) != 2 or proof["source"] != card["source"]:
            raise ValueError("incomplete source admission evidence")
        direct = path.with_name(f"direct-{path.stem.removeprefix('accepted-')}-1.json")
        if sha_file(direct) != card["direct_sha256"]:
            raise ValueError("source continuation changed")
        for trial, admitted in enumerate(proof["metrics"]):
            run = read(root / card["source_replays"] / f"raw-{trial}.json")
            rows = run["trace"]["steps"]
            start = len(anchor["prefix"])
            if (run["spec"] != specs[card["source"]]
                    or [admitted[a]["success"] for a in ("silent", "direct", "remind")] != [0, 1, 1]
                    or run["metrics"]["success"] != 1
                    or run["metrics"]["actor_calls"] > card["source_cap"]
                    or run["metrics"]["utility"] <= admitted["silent"]["utility"]
                    or rows[:start] != anchor["prefix"] or rows[start]["state"] != anchor["state"]
                    or run["extra_draft"] != anchor["draft"]):
                raise ValueError("source reminder could not be verified")


def failure_interval(rows, seed):
    by_template = defaultdict(lambda: [0, 0])
    for template, row in rows:
        s, m = row["silent"]["success"], row["episodic"]["success"]
        pair = by_template[template]
        pair[0] += int(s and not m)
        pair[1] += s
    counts = np.asarray(list(by_template.values()), dtype=int)
    picks = np.random.default_rng(seed).integers(len(counts), size=(3000, len(counts)))
    simulated = counts[picks].sum(axis=1)
    valid = simulated[:, 1] > 0
    point = counts.sum(axis=0)
    return {
        "harmed": int(point[0]), "silent_successes": int(point[1]),
        "rate": float(point[0] / point[1]) if point[1] else None,
        "template_cluster_95": (
            np.quantile(simulated[valid, 0] / simulated[valid, 1], [.025, .975]).tolist()
            if valid.any() else None
        ),
    }


def harmed_cases(rows, full_by_id):
    cases = []
    for _, row in rows:
        s, n, m = (row[arm] for arm in ("silent", "neutral", "episodic"))
        if not s["success"] or m["success"]:
            continue
        case = row["match"]
        if case is None:
            raise ValueError("an untriggered case cannot regress")
        full = full_by_id[row["spec"]]
        cases.append({
            "spec": row["spec"], "step": case["step"], "source": case["source"],
            "strict_contract": case["strict_contract"], "neutral_success": n["success"],
            "success_with_full_call_budget": full["episodic"]["success"],
            "original_action": case["original"], "neutral_first_action": case["neutral_first_action"],
            "episodic_first_action": case["episodic_first_action"],
            "delta_tokens": m["tokens"] - s["tokens"], "delta_seconds": m["seconds"] - s["seconds"],
        })
    return cases


def redraft_contrast(rows):
    values = Counter()
    for _, row in rows:
        s, n, m = (row[arm] for arm in ("silent", "neutral", "episodic"))
        values.update(
            memory_loses_to_neutral=int(n["success"] and not m["success"]),
            memory_wins_over_neutral=int(m["success"] and not n["success"]),
            neutral_minus_silent_tokens=n["tokens"] - s["tokens"],
            memory_minus_neutral_tokens=m["tokens"] - n["tokens"],
            neutral_minus_silent_seconds=n["seconds"] - s["seconds"],
            memory_minus_neutral_seconds=m["seconds"] - n["seconds"],
        )
    return dict(values)


def audit(root, source, runtime, development):
    freeze = read(root / "transfer-freeze.json")
    baseline = Path(freeze["baseline"])
    files = [
        (root / "transfer-bank.json", freeze["bank_sha256"]),
        (baseline / "actor.json", freeze["actor_sha256"]),
        (baseline / "manifest.json", freeze["manifest_sha256"]),
        *[(runtime / name, sha) for name, sha in freeze["code"].items()],
    ]
    if any(sha_file(path) != sha for path, sha in files):
        raise ValueError("frozen study input changed")
    manifest = read(baseline / "manifest.json")
    bank = read(root / "transfer-bank.json")
    if sha_file(source / "bank.json") != bank["construction_bank_sha256"]:
        raise ValueError("construction bank changed")
    check_sources(root, source, bank, manifest)
    groups = {"development": len(manifest["splits"]["development"])} if development else {
        "valid_seen": 140, "valid_unseen": 134,
    }
    cards = {card["id"] for card in bank["cards"]}
    budgets = freeze["token_budgets"]
    scores = {z: defaultdict(list) for z in ("calls", *budgets)}
    args = SimpleNamespace(out=root)
    for group, expected in groups.items():
        specs = manifest["splits"][group]
        if len(specs) != expected or len({spec["id"] for spec in specs}) != expected:
            raise ValueError("incomplete or duplicate cohort")
        for spec in specs:
            natural = read(receipt(baseline, spec))
            paired = read(receipt(root, spec, "paired"))
            branches = audit_pair(args, spec, natural, paired, cards)
            template = spec["template"]
            scores["calls"][group].append((template, paired))
            for name, budget in budgets.items():
                rows = score_tokens(natural, paired["match"], branches, budget)
                if any(rows[arm]["tokens"] > budget for arm in freeze["arms"]):
                    raise ValueError("token budget exceeded")
                rows["spec"] = spec["id"]
                scores[name][group].append((template, rows))
    aggregate = {}
    intervals = {}
    scope_breakdown = {}
    regressions = {}
    redrafting = {}
    full_by_id = {row["spec"]: row for group in groups for _, row in scores["calls"][group]}
    for setting, runs in scores.items():
        runs["overall"] = [p for group in groups for p in runs[group]]
        aggregate[setting] = {g: summary([row for _, row in rows]) for g, rows in runs.items()}
        intervals[setting] = {g: failure_interval(rows, manifest["seed"]) for g, rows in runs.items()}
        match_rows = [row for _, row in runs["overall"] if row["match"]]
        scope_breakdown[setting] = {
            "strict_contract_holds": summary([row for row in match_rows if row["match"]["strict_contract"]]),
            "outside_strict_contract": summary([row for row in match_rows if not row["match"]["strict_contract"]]),
        }
        regressions[setting] = harmed_cases(runs["overall"], full_by_id)
        redrafting[setting] = {g: redraft_contrast(rows) for g, rows in runs.items()}
    stage = "development" if development else "summary"
    if aggregate != read(root / f"{stage}.json"):
        raise ValueError("reported aggregates do not match the records")
    result = {"tasks": sum(groups.values()), "verified_source_memories": len(cards),
              "summary": aggregate, "failure_intervals": intervals, "scope_breakdown": scope_breakdown,
              "harmed_cases": regressions, "redraft_contrast": redrafting}
    save(root / f"audited-{stage}.json", result)
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "source", "runtime"):
        cli.add_argument(f"--{field}", type=Path, required=True)
    cli.add_argument("--development", action="store_true")
    args = cli.parse_args()
    result = audit(args.root, args.source, args.runtime, args.development)
    print("TRANSFER AUDITED", result["tasks"], result["verified_source_memories"])
    for name, groups in result["failure_intervals"].items():
        print(name, groups["overall"])
