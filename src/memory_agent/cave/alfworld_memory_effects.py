"""Rebuild legacy paired features with frozen memory identity and extra development games."""

import argparse
from pathlib import Path
from types import SimpleNamespace

from memory_agent.cave.alfworld_env import sha_file
from memory_agent.cave.alfworld_gate_refit import paired_rows
from memory_agent.cave.alfworld_gate_tune import context_features
from memory_agent.cave.alfworld_study_rollout import receipt
from memory_agent.cave.alfworld_transfer_study import audit_pair, score_tokens
from memory_agent.cave.records import read, save


def attach(natural, match, card, vocab, budget, order):
    return [*context_features(natural, match, card, vocab, budget),
            *(int(card["id"] == key) for key in order)]


def extra_pairs(root, study, transfer, bank, vocab, order):
    registry = read(root / "manifest.json")
    proof = read(root / "probe-freeze.json")
    if (proof["study_sha256"] != sha_file(study / "transfer-freeze.json")
            or proof["cohort_sha256"] != sha_file(root / "manifest.json")):
        raise ValueError("extra development cohort does not match the same frozen intervention")
    specs = {r["id"]: r for r in registry["splits"]["development"]}
    combined = {"train": [], "validation": []}
    for split, games in registry["heads_splits"].items():
        for game in games:
            spec = specs[game]
            natural = read(receipt(root, spec))
            paired = read(receipt(root, spec, "paired"))
            branches = audit_pair(SimpleNamespace(out=root), spec, natural, paired, set(bank))
            for setting, budget in [("calls", None), *transfer["token_budgets"].items()]:
                source = paired if budget is None else score_tokens(natural, paired["match"], branches, budget)
                match = source["match"]
                x = attach(natural, match, bank[match["card"]], vocab, budget, order) if match else None
                s, r = source["silent"], source["episodic"]
                combined[split].append(
                    {"game": game, "template": spec["template"], "setting": setting, "features": x,
                     "silent": s, "remind": r, "D": r["utility"] - s["utility"],
                     "H": int(s["success"] and not r["success"]), "extra": r["tokens"] - s["tokens"]}
                )
    return combined


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "refit", "study", "supplement"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args()
    transfer_file = args.study / "transfer-freeze.json"
    transfer = read(transfer_file)
    cohort = read(args.refit / "cohort.json")
    base = Path(transfer["baseline"])
    bank_file = args.study / "transfer-bank.json"
    if transfer["bank_sha256"] != sha_file(bank_file) or cohort["transfer_sha256"] != sha_file(transfer_file):
        raise ValueError("the archives refer to different reminder interventions")
    bank = {r["id"]: r for r in read(bank_file)["cards"]}
    order = sorted(bank)
    vocab = set(read(base / "ranker.json")["vocabulary"])
    rows = paired_rows(args.study, transfer, read(base / "manifest.json"), cohort,
                       lambda a, b, c, d, e: attach(a, b, c, d, e, order))
    extra = extra_pairs(args.supplement, args.study, transfer, bank, vocab, order)
    for split in rows:
        rows[split].extend(extra[split])
        for row in rows[split]:
            row["R"] = int(row["remind"]["success"] and not row["silent"]["success"])
    if {r["template"] for r in rows["train"]} & {r["template"] for r in rows["validation"]}:
        raise ValueError("memory effect development cohorts overlap")
    save(args.out / "collector-freeze.json", {"selection": "legacy-wide-persistent",
                                              "train_archive_sha256": sha_file(args.refit / "cohort.json"),
                                              "supplement_sha256": sha_file(args.supplement / "manifest.json"),
                                              "transfer_sha256": sha_file(transfer_file), "memory_order": order})
    save(args.out / "paired-labels.json", rows)
    print("MEMORY_EFFECT_COHORT", {k: len({r["game"] for r in v}) for k, v in rows.items()}, flush=True)


if __name__ == "__main__":
    main()
