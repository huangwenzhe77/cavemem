"""Prospective two-card mini experiment; the published single-card protocol remains replayable."""

import argparse
import math
import random
import statistics
import tempfile
from pathlib import Path
from types import SimpleNamespace

from . import development_card
from .branches import branch, load_actor, revision_fits
from .experiment import archive_digest, audit_aa, match, open_bank
from .pilot import check_run, load_plan, shadow_cost
from .pilot_report import cluster_mean, design_intervals, mean, rate, reconcile, summarize
from .records import digest, read, save
from .replay import protocol_digest, require, restore, selected_turn


ROUTER = "mini-family-router-v2.1"


def verified_cards(plan, parent_bank, package, previous_plan):
    parent_id, old_cards, old_inputs = open_bank(parent_bank)
    require(len(old_cards) == 1, "expected one verified ascending card")
    desc = read(package / "development-v2/card.json")
    # Independently rebind this export to both discovery/confirmation arms,
    # score receipts, the source task manifest and the exact reviewed proposal.
    with tempfile.TemporaryDirectory() as tmp:
        rebuilt = Path(tmp) / "card"
        development_card.export(package, rebuilt)
        require(
            desc == read(rebuilt / "card.json"), "descending development export diverged from evidence"
        )
    bank_id = digest(
        {
            "schema": "mini-prospective-bank-v2",
            "parent_bank_id": parent_id,
            "descending_memory_id": desc["memory_id"],
        }
    )
    require(
        parent_id == plan["parent_bank_id"]
        and desc["memory_id"] == plan["descending_memory_id"]
        and bank_id == plan["bank_id"],
        "preregistered bank mismatch",
    )
    excluded = set(old_inputs)
    excluded.update(digest(t["input"]) for t in read(package / "source/manifest.json")["tasks"])
    prior = read(previous_plan)
    require(
        digest(prior) == plan["previous_plan_digest"]
        and prior["bank_id"] == parent_id
        and len(prior["tasks"]) == 32,
        "the preceding 32-episode pilot manifest must be supplied",
    )
    excluded.update(digest(t["task"]["input"]) for t in prior["tasks"])
    inputs = [digest(t["task"]["input"]) for c in plan["cohorts"] for t in c["tasks"]]
    require(
        len(set(inputs)) == len(inputs) and not excluded.intersection(inputs),
        "new target duplicates a source, another target, or a previous pilot input",
    )
    return {"ascending": old_cards[0], "descending": desc}


def route(cards, spec, pre, draft):
    if spec["family"] != "descending":
        return match([cards["ascending"]], spec, pre, draft)
    card = cards["descending"]
    rho = int(development_card.applicable(spec, pre["state"]))
    e = development_card.exposure(draft["a0"])
    return {
        "status": "inapplicable"
        if not rho
        else {0: "avoided", 1: "exposed", None: "unknown_exposure"}[e],
        "memory_id": card["memory_id"],
        "v": 1,
        "rho": rho,
        "e": e,
        "rule": card["contract"],
        "observed_ops": [c["keystrokes"].strip() for c in draft["a0"]["commands"]],
        "reminder": card["candidate"]["reminder"],
    }


def schedule(run, manifest, rows, cards, matcher=route):
    """Retrieval and timing see the same prefix and never consult branch outcomes."""
    decisions = {}

    def record(cards, spec, pre, draft):
        decision = matcher(cards, spec, pre, draft)
        decisions[pre["checkpoint_id"]] = {
            **decision,
            "checkpoint_id": pre["checkpoint_id"],
            "pre_digest": digest(pre),
            "a0_digest": draft["a0_digest"],
        }
        return decision

    shadows = shadow_cost(run, manifest, rows, cards, matcher=record)
    ledger = []
    for row in rows:
        turn = selected_turn(run, manifest, row)
        d = (
            decisions[f"{row['episode_id']}/{turn:03d}"]
            if turn is not None
            else {"status": "never_reached", "memory_id": None, "v": None, "rho": None, "e": None}
        )
        ledger.append(
            {
                "episode_id": row["episode_id"],
                "main_opportunity": row["main_opportunity"],
                "turn": turn,
                "q": row["q"],
                **d,
            }
        )
    return ledger, shadows


def check_free(path, run, manifest, row, decision, plan, actor):
    signatures = []
    for side in ("a", "b"):
        live = branch(
            run,
            manifest,
            row,
            decision["turn"],
            actor,
            "free",
            "",
            plan["seed"],
            plan["total_tokens"],
            plan["total_tools"],
        )
        save(path / f"free-{side}.json", live)
        if live["score"]["reward"] is None or not live["requests"]:
            return {"free_status": "unavailable", "free_error": live["termination"]}
        signatures.append(
            digest(
                [
                    [r["generation"]["text"] for r in live["requests"]],
                    live["events"],
                    live["score"]["reward"],
                ]
            )
        )
    return {
        "free_status": "agree" if signatures[0] == signatures[1] else "drift",
        "free_anchor": decision["turn"],
        "free_digests": signatures,
    }


def arm_record(pdir, pair_id, run, manifest, row, d, plan, actor, arm, seed):
    live = branch(
        run,
        manifest,
        row,
        d["turn"],
        actor,
        arm,
        d["reminder"],
        seed,
        plan["total_tokens"],
        plan["total_tools"],
    )
    ref = f"pairs/{pair_id}/{arm}.json"
    save(pdir / ref, live)
    u = live["usage"]
    seconds = u["common_actor_seconds"] + u["new_actor_seconds"] + u["new_tool_seconds"]
    penalty = (
        plan["lambda_tokens"] * (u["input_tokens"] + u["output_tokens"]) / 1000
        + plan["lambda_tool"] * u["tool_calls"]
        + plan["lambda_time"] * seconds / 60
    )
    y = live["score"]["reward"]
    return {
        "file": ref,
        "digest": digest(live),
        "Y": y,
        "utility": None if y is None else y - penalty,
        "cost_penalty": penalty,
        "accounted_seconds": seconds,
        "revision_calls": live["revision_calls"],
        "usage": u,
    }


def collect_pairs(pdir, run, manifest, rows, ledger, checks, plan, policy, actor):
    pairs = []
    for row, d, check in zip(rows, ledger, checks, strict=True):
        if d.get("rho") != 1:
            continue
        eid = row["episode_id"]
        require(eid == d["episode_id"] == check["episode_id"], "check/decision identity mismatch")
        if (
            check.get("tape_status") != "pass"
            or check.get("free_status") != "agree"
            or (
                check.get("free_anchor") != d["turn"]
                or check["archive_digest"] != archive_digest(run, row)
            )
        ):
            pairs.append({"episode_id": eid, "status": "restore_unapproved"})
            continue
        pre, draft, _ = restore(run, row, d["turn"])
        require(
            digest(pre) == d["pre_digest"] and draft["a0_digest"] == d["a0_digest"],
            "cached action changed",
        )
        if pre["used_tool_calls"] > plan["total_tools"] or not revision_fits(
            manifest, row, d["turn"], actor, pre, draft, d["reminder"], plan["total_tokens"]
        ):
            pairs.append({"episode_id": eid, "status": "insufficient_revision_budget"})
            continue
        for rep in range(plan["replicates"]):
            pair_id = digest([digest(policy), d, rep])
            order = ["silent", "remind"]
            random.Random("order:" + pair_id).shuffle(order)
            seed = random.Random(f"coupling:{plan['seed']}:{manifest['split']}:{eid}:{rep}").randrange(
                2**31
            )
            arms = {
                arm: arm_record(pdir, pair_id, run, manifest, row, d, plan, actor, arm, seed)
                for arm in order
            }
            s, r = arms["silent"], arms["remind"]
            result = {
                "pair_id": pair_id,
                "episode_id": eid,
                "checkpoint_id": d["checkpoint_id"],
                "a0_digest": d["a0_digest"],
                "memory_id": d["memory_id"],
                "replicate": rep,
                "seed": seed,
                "order": order,
                "status": "incomplete",
                "arms": arms,
            }
            if s["Y"] is not None and r["Y"] is not None:
                result.update(
                    status="complete",
                    difference=r["utility"] - s["utility"],
                    regression=int(s["Y"] == 1 and r["Y"] == 0),
                    rescue=int(s["Y"] == 0 and r["Y"] == 1),
                )
            pairs.append(result)
            save(pdir / "pairs.json", pairs)
    save(pdir / "pairs.json", pairs)
    return pairs


def get_plan(root):
    plan, pid = load_plan(root / "preregistration.json")
    require(
        plan["cohort_version"] == "prospective-v2"
        and plan["router_version"] == ROUTER
        and plan["protocol"] == protocol_digest(),
        "wrong cohort, router, or frozen engine",
    )
    require(
        plan["replicates"] > 0
        and plan["total_tokens"] > 0
        and plan["total_tools"] > 0
        and all(
            math.isfinite(plan[k]) and plan[k] >= 0
            for k in ("lambda_tokens", "lambda_time", "lambda_tool")
        ),
        "invalid experiment budget or utility weights",
    )
    return plan, pid


def audit(args):
    plan, pid = get_plan(args.root)
    cards = verified_cards(plan, args.parent_bank, args.source_package, args.previous_plan)
    run = args.root / args.split
    manifest, rows = check_run(plan, args.split, run)
    out = args.root / ("audit-" + args.split)
    audit_run(out, run, manifest, rows, plan, pid, cards)


def audit_run(out, run, manifest, rows, plan, pid, cards, matcher=route):
    """Run the same audited forks on a caller-verified cohort and registered matcher."""
    out.mkdir(parents=True, exist_ok=False)
    pdir = out / "pairs"
    ledger, shadows = schedule(run, manifest, rows, cards, matcher)
    save(pdir / "matches.json", ledger)
    save(
        out / "shadow.json",
        {
            "plan_id": pid,
            "run_id": manifest["run_id"],
            "rows": shadows,
            "measurement": "offline timed family routing and predicates after archive load; no additional actor or tool",
        },
    )
    # Fixed-tape restoration runs for every predeclared episode. Only an
    # applicable real decision can incur additional live-model A/A or pairing.
    audit_aa(SimpleNamespace(run=run, out=out / "aa", free=False))
    aa = read(out / "aa/aa.json")
    del aa["audit_id"]
    actor = None
    for row, d, c in zip(rows, ledger, aa["checks"], strict=True):
        if d.get("rho") == 1 and c["tape_status"] == "pass":
            if actor is None:
                actor = load_actor(manifest)
            c.update(check_free(out / "aa" / row["episode_id"], run, manifest, row, d, plan, actor))
    aa.update(plan_id=pid, router_version=plan["router_version"])
    audit_id = digest(aa)
    save(out / "aa/aa.json", {**aa, "audit_id": audit_id})
    policy = {
        "schema": 2,
        "protocol": protocol_digest(),
        "router_version": plan["router_version"],
        "bank_id": plan["bank_id"],
        "parent_bank_id": plan["parent_bank_id"],
        "descending_memory_id": plan["descending_memory_id"],
        "plan_id": pid,
        "audit_id": audit_id,
        "run_id": manifest["run_id"],
        "split": manifest["split"],
        "cohort": "natural",
        "replicates": plan["replicates"],
        "seed": plan["seed"],
        "selection": "one scheduled opportunity; route by family; retain unknown exposure",
        "limits": {
            "actor_calls": manifest["max_steps"],
            "steps": manifest["max_steps"],
            "total_tokens": plan["total_tokens"],
            "tool_calls": plan["total_tools"],
        },
        "utility_weights": {
            "per_1000_tokens": plan["lambda_tokens"],
            "per_tool": plan["lambda_tool"],
            "per_minute": plan["lambda_time"],
        },
        "comparison": "matched deterministic fork with shared suffix seed; no neutral comparator or arm propensity",
    }
    save(pdir / "protocol.json", policy)
    pairs = collect_pairs(pdir, run, manifest, rows, ledger, aa["checks"], plan, policy, actor)
    print(
        manifest["split"],
        len(rows),
        "episodes;",
        sum(p["status"] == "complete" for p in pairs),
        "complete pairs",
    )


def report(args):
    plan, pid = get_plan(args.root)
    strata = {}
    rows = []
    for spec in plan["cohorts"]:
        policy = read(args.root / ("audit-" + spec["split"]) / "pairs/protocol.json")
        require(
            all(
                policy[k] == plan[k]
                for k in ("router_version", "parent_bank_id", "descending_memory_id")
            )
            and policy["plan_id"] == pid,
            "wrong prospective router lineage",
        )
        group = reconcile(args.root, plan, pid, spec)
        rows.extend(group)
        strata[spec["split"]] = summarize(group)
    by_exposure = {}
    for status in ("avoided", "exposed", "unknown_exposure"):
        subset = [r for r in rows if r["match"].get("rho") == 1 and r["match"]["status"] == status]
        by_exposure[status] = summarize(subset)
    applicable_rows = [r for r in rows if r["match"].get("rho") == 1]
    old_count = sum(r["stratum"] == "controlled_transfer" for r in applicable_rows)
    known = sum(r["match"]["e"] in (0, 1) for r in applicable_rows)
    avoided = by_exposure["avoided"]["episodes"]
    success_cost = {}
    for metric in ("total_tokens", "actor_calls", "tool_calls", "accounted_seconds", "utility"):
        success_cost[metric] = cluster_mean(
            [mean([p["delta"][metric] for p in r["pairs"] if p["cell"] == "both_pass"]) for r in rows]
        )
    values = [
        r["mean_paired_delta"]["utility"] for r in rows if r["mean_paired_delta"]["utility"] is not None
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "episode-index.json", rows)
    save(
        args.out / "summary.json",
        {
            "schema": 2,
            "plan_id": pid,
            "bank_id": plan["bank_id"],
            "router_version": ROUTER,
            "strata": strata,
            "pooled": summarize(rows),
            "applicable_by_action_exposure": by_exposure,
            "fixed_mixture_uncertainty": design_intervals(rows, strata),
            "coverage_on_identical_scheduled_tasks": {
                "denominator": len(rows),
                "ascending_only_applicable": old_count,
                "two_card_applicable": len(applicable_rows),
                "increment": len(applicable_rows) - old_count,
                "method": "Ascending routes reuse the frozen match function verbatim. The old ascending-only bank "
                "returns no candidate on every descending task. This is an applicability comparison only.",
            },
            "action_exposure": {
                "avoided_among_all_applicable": rate(avoided, len(applicable_rows)),
                "avoided_among_known_applicable": rate(avoided, known),
                "unknown_among_applicable": rate(
                    by_exposure["unknown_exposure"]["episodes"], len(applicable_rows)
                ),
            },
            "both_success_cost_deltas": success_cost,
            "precision": {
                "complete_contrast_episodes": len(values),
                "net_utility_sd": statistics.stdev(values) if len(values) > 1 else None,
                "scope": "Conditional, restricted to completed pairs in two templates. Replicate seeds do not increase "
                "the independent task count. Event-rate intervals are in the pooled and stratum reports.",
            },
            "interpretation": "One prescheduled opportunity per fresh task. Conditional effects require independently "
            "scored complete pairs. Unknown exposure is separate. The treatment combines a reminder and one "
            "redraft. Shadow overhead covers in-memory matching only. This comparison tests fixed policies.",
        },
    )
    print(args.out / "summary.json")


def main():
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--split", required=True, choices=("target_train", "controlled_transfer"))
    for name in ("root", "parent-bank", "source-package", "previous-plan"):
        a.add_argument("--" + name, required=True, type=Path)
    r = sub.add_parser("report")
    r.add_argument("--root", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    (audit if args.command == "audit" else report)(args)


if __name__ == "__main__":
    main()
