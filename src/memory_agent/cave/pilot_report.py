"""Reconcile all preregistered mini episodes and paired observations by task."""

import argparse
import math
import random
import statistics
from collections import Counter
from pathlib import Path

from .pilot import check_run, load_plan
from .records import digest, read, save
from .replay import require


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def rate(k, n, z=1.95996398454):
    if not n:
        return {"numerator": k, "denominator": 0, "rate": None, "wilson95": None}
    p = k / n
    scale = 1 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return {
        "numerator": k,
        "denominator": n,
        "rate": p,
        "wilson95": [max(0.0, center - width), min(1.0, center + width)],
    }


def cluster_mean(values):
    values = [v for v in values if v is not None]
    ci = None
    if len(values) >= 2:
        rng = random.Random(97491)
        means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
        ci = [means[49], means[1949]]
    return {"n_episodes": len(values), "mean": mean(values), "episode_bootstrap95": ci}


def indexed(records):
    result = {r["episode_id"]: r for r in records}
    require(len(records) == len(result), "duplicate task IDs")
    return result


CELLS = ("both_pass", "both_fail", "rescue", "regression")
DELTA_KEYS = (
    "success",
    "utility",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "actor_calls",
    "tool_calls",
    "accounted_seconds",
    "new_actor_seconds",
    "new_tool_seconds",
    "branch_wall_seconds",
    "cost_usd",
)


def outcome_cell(silent, remind):
    return {(1, 1): "both_pass", (0, 0): "both_fail", (0, 1): "rescue", (1, 0): "regression"}[
        silent, remind
    ]


def paired_obs(row, pairs_dir, plan):
    arms = {}
    for name in ("silent", "remind"):
        ref = row["arms"][name]
        path = Path(ref["file"])
        require(path.as_posix() == f"pairs/{row['pair_id']}/{name}.json", "wrong pair evidence path")
        live = read(pairs_dir / path)
        require(
            digest(live) == ref["digest"]
            and live["a0_digest"] == row["a0_digest"]
            and live["arm"] == name
            and live["score"]["reward"] == ref["Y"]
            and live["usage"] == ref["usage"],
            "branch evidence mismatch",
        )
        u = ref["usage"]
        seconds = u["common_actor_seconds"] + u["new_actor_seconds"] + u["new_tool_seconds"]
        tokens = u["input_tokens"] + u["output_tokens"]
        penalty = (
            plan["lambda_tokens"] * tokens / 1000
            + plan["lambda_tool"] * u["tool_calls"]
            + plan["lambda_time"] * seconds / 60
        )
        y = ref["Y"]
        if y is not None:
            require(
                math.isclose(ref["utility"], y - penalty, abs_tol=1e-10), "utility differs from raw cost"
            )
        arms[name] = {
            **u,
            "success": y,
            "total_tokens": tokens,
            "accounted_seconds": seconds,
            "utility": ref["utility"],
            "score_status": live["score"]["status"],
            "termination": live["termination"],
        }
    s, r = arms["silent"], arms["remind"]
    yes = s["success"] is not None and r["success"] is not None
    require((row["status"] == "complete") == yes, "completion status disagrees with verifier")
    changes = {
        k: r[k] - s[k] if yes and r[k] is not None and s[k] is not None else None for k in DELTA_KEYS
    }
    return {
        "pair_id": row["pair_id"],
        "replicate": row["replicate"],
        "arms": arms,
        "cell": outcome_cell(s["success"], r["success"]) if yes else None,
        "delta": changes,
    }


def reconcile(root, plan, pid, spec, run_reader=check_run):
    split = spec["split"]
    run = root / split
    audit = root / ("audit-" + split)
    manifest, rows = run_reader(plan, split, run)
    shadow = read(audit / "shadow.json")
    aa = read(audit / "aa/aa.json")
    pdir = audit / "pairs"
    study = read(pdir / "protocol.json")
    require(
        aa["audit_id"] == digest({k: v for k, v in aa.items() if k != "audit_id"}), "corrupt A/A audit"
    )
    require(
        shadow["plan_id"] == pid
        and manifest["run_id"] == shadow["run_id"] == aa["run_id"] == study["run_id"]
        and aa["manifest_digest"] == digest(manifest)
        and study["audit_id"] == aa["audit_id"]
        and plan["bank_id"] == study["bank_id"]
        and study["seed"] == plan["seed"]
        and plan["replicates"] == study["replicates"]
        and study["cohort"] == "natural",
        "pilot, bank, audit, or schedule lineage mismatch",
    )
    require(
        study["limits"]
        == {
            "actor_calls": plan["max_steps"],
            "steps": plan["max_steps"],
            "total_tokens": plan["total_tokens"],
            "tool_calls": plan["total_tools"],
        }
        and study["utility_weights"]
        == {
            "per_1000_tokens": plan["lambda_tokens"],
            "per_tool": plan["lambda_tool"],
            "per_minute": plan["lambda_time"],
        },
        "budget or price drift",
    )
    matches = indexed(read(pdir / "matches.json"))
    audits = indexed(aa["checks"])
    shadows = indexed(shadow["rows"])
    keys = {r["episode_id"] for r in rows}
    require(set(matches) == set(audits) == set(shadows) == keys, "incomplete task denominator")
    by_episode = {k: [] for k in keys}
    ids = set()
    for evidence in read(pdir / "pairs.json"):
        require(evidence["episode_id"] in keys, "unknown intervention episode")
        if "pair_id" in evidence:
            require(evidence["pair_id"] not in ids, "duplicate pair")
            ids.add(evidence["pair_id"])
        by_episode[evidence["episode_id"]].append(evidence)
    output = []
    for row in rows:
        eid = row["episode_id"]
        m, a, s = matches[eid], audits[eid], shadows[eid]
        require(
            m["q"] == row["q"]
            and m["turn"] == row["main_turn_id"] == a.get("selected_turn")
            and m["main_opportunity"] == row["main_opportunity"]
            and s["status"] == m["status"]
            and s["run_id"] == manifest["run_id"],
            "task opportunity or match changed",
        )
        observations = []
        reasons = []
        for p in by_episode[eid]:
            require(m.get("rho") == 1, "inapplicable checkpoint has a branch")
            if "pair_id" in p:
                require(
                    p["checkpoint_id"] == m["checkpoint_id"]
                    and p["memory_id"] == m["memory_id"]
                    and p["a0_digest"] == m["a0_digest"],
                    "match/branch mismatch",
                )
                observations.append(paired_obs(p, pdir, plan))
            else:
                reasons.append(p["status"])
        require(
            len({p["replicate"] for p in observations}) == len(observations)
            and all(0 <= p["replicate"] < plan["replicates"] for p in observations),
            "unexpected repeat or duplicate episode/replicate",
        )
        if m.get("rho") == 1 and len(observations) < plan["replicates"] and not reasons:
            reasons.append("expected_pair_missing")
        completed = [p for p in observations if p["cell"] is not None]
        bare = row["verifier"]
        require(read(run / eid / "verdict.json") == bare, "baseline verdict changed")
        paired_delta = {k: mean([p["delta"][k] for p in completed]) for k in DELTA_KEYS}
        noop = m.get("rho") != 1 or "insufficient_revision_budget" in reasons
        schedule_delta = {}
        for key in (
            "success",
            "total_tokens",
            "actor_calls",
            "tool_calls",
            "accounted_seconds",
            "utility",
        ):
            value = 0 if noop else paired_delta[key] if len(completed) == plan["replicates"] else None
            if value is not None and key == "accounted_seconds":
                value += s["match_seconds"]
            if value is not None and key == "utility":
                value -= plan["lambda_time"] * s["match_seconds"] / 60
            schedule_delta[key] = value
        output.append(
            {
                "stratum": split,
                "episode_id": eid,
                "seed": row["seed"],
                "task_digest": digest(row["task"]),
                "q": row["q"],
                "main_opportunity": row["main_opportunity"],
                "turn": m["turn"],
                "reached": row["main_reached"],
                "match": m,
                "tape_aa": a.get("tape_status"),
                "free_aa": a.get("free_status"),
                "bare_score": bare,
                "shadow": s,
                "pairs": observations,
                "unpaired_reasons": reasons,
                "complete_pairs": len(completed),
                "mean_paired_delta": paired_delta,
                "scheduled_delta_vs_bare": schedule_delta,
            }
        )
    return output


def summarize(rows):
    n = len(rows)
    reached = sum(r["reached"] for r in rows)
    candidates = sum(bool(r["match"].get("memory_id")) for r in rows)
    eligible = sum(r["match"].get("rho") == 1 for r in rows)
    pairs = [p for r in rows for p in r["pairs"]]
    complete = [p for p in pairs if p["cell"] is not None]
    counts = Counter(p["cell"] for p in complete)
    bare = [r["bare_score"]["reward"] for r in rows]
    both = [p for p in complete if p["cell"] == "both_pass"]
    bins = Counter(r["match"]["status"] for r in rows)
    missing_cases = Counter(reason for r in rows for reason in r["unpaired_reasons"])
    # For the scheduled strategy, a no-op leaves the observed task outcome
    # unchanged. A missing intervention has unknown potential outcomes.
    policy = []
    for r in rows:
        if r["match"].get("rho") != 1 or "insufficient_revision_budget" in r["unpaired_reasons"]:
            policy.append(r["bare_score"]["reward"])
        elif r["pairs"] and len(r["pairs"]) == r["complete_pairs"] and not r["unpaired_reasons"]:
            policy.append(mean([p["arms"]["remind"]["success"] for p in r["pairs"]]))
        else:
            policy.append(None)
    observed_policy = [v for v in policy if v is not None]
    matched_tasks = sum(bool(r["complete_pairs"]) for r in rows)
    return {
        "episodes": n,
        "reach": rate(reached, n),
        "memory_candidate": rate(candidates, n),
        "applicable": rate(eligible, n),
        "candidate_given_reached": rate(candidates, reached),
        "applicable_given_reached": rate(eligible, reached),
        "match_states": dict(bins),
        "tape_restored": rate(sum(r["tape_aa"] == "pass" for r in rows), n),
        "free_aa_agree_given_applicable": rate(
            sum(r["free_aa"] == "agree" and r["match"].get("rho") == 1 for r in rows), eligible
        ),
        "bare_pass_at_1_observed": rate(sum(x == 1 for x in bare), sum(x is not None for x in bare)),
        "bare_missing": rate(bare.count(None), n),
        "scheduled_remind_strategy": {
            "observed_episodes": len(observed_policy),
            "success_mean_observed": mean(observed_policy),
            "success_full_population_bounds": [
                sum(observed_policy) / n,
                (sum(observed_policy) + policy.count(None)) / n,
            ]
            if n
            else None,
        },
        "attempted_pairs": len(pairs),
        "complete_pairs": len(complete),
        "episodes_with_complete_pairs": matched_tasks,
        "incomplete_given_attempted": rate(len(pairs) - len(complete), len(pairs)),
        "no_complete_pair_given_applicable": rate(eligible - matched_tasks, eligible),
        "missing_reason_counts": dict(missing_cases),
        "four_cells": {cell: counts[cell] for cell in CELLS},
        "conditional_cell_rates": {cell: rate(counts[cell], len(complete)) for cell in CELLS},
        "both_pass_slower": rate(sum(p["delta"]["accounted_seconds"] > 0 for p in both), len(both)),
        "both_pass_more_tokens": rate(sum(p["delta"]["total_tokens"] > 0 for p in both), len(both)),
        "paired_delta_by_episode": {
            k: cluster_mean([r["mean_paired_delta"][k] for r in rows]) for k in DELTA_KEYS
        },
        "scheduled_delta_vs_bare": {
            k: cluster_mean([r["scheduled_delta_vs_bare"][k] for r in rows])
            for k in (
                "success",
                "total_tokens",
                "actor_calls",
                "tool_calls",
                "accounted_seconds",
                "utility",
            )
        },
        "shadow_match_seconds_per_task": cluster_mean([r["shadow"]["match_seconds"] for r in rows]),
        "bare_mean_per_task": {
            k: mean([r["shadow"][k] for r in rows])
            for k in (
                "bare_actor_seconds",
                "bare_actor_calls",
                "bare_input_tokens",
                "bare_output_tokens",
                "bare_tool_calls",
            )
        },
        "shadow_increment_total": {
            "actor_calls": sum(r["shadow"]["shadow_additional_actor_calls"] for r in rows),
            "tokens": sum(r["shadow"]["shadow_additional_tokens"] for r in rows),
            "tool_calls": sum(r["shadow"]["shadow_additional_tool_calls"] for r in rows),
            "match_calls": sum(r["shadow"]["retrieval_calls"] for r in rows),
            "match_seconds": sum(r["shadow"]["match_seconds"] for r in rows),
        },
    }


def design_intervals(rows, strata):
    """Approximate simultaneous stratum bounds for the fixed equal mixture."""
    result = {}
    for key in (
        "reach",
        "memory_candidate",
        "applicable",
        "bare_missing",
        "bare_pass_at_1_observed",
        "tape_restored",
    ):
        bounds = []
        for group in strata.values():
            endpoint = group[key]
            interval = rate(endpoint["numerator"], endpoint["denominator"], z=2.24140273)["wilson95"]
            if interval is None:
                bounds = []
                break
            bounds.append(interval)
        result[key] = [mean([b[i] for b in bounds]) for i in range(2)] if bounds else None
    split_rows = [[r for r in rows if r["stratum"] == s] for s in strata]
    # No loss of template weights as outcomes are resampled within each template.
    scheduled = {}
    for key in ("success", "total_tokens", "actor_calls", "tool_calls", "accounted_seconds", "utility"):
        groups = [[r["scheduled_delta_vs_bare"][key] for r in members] for members in split_rows]
        if any(len(g) < 2 or any(v is None for v in g) for g in groups):
            scheduled[key] = None
            continue
        rng = random.Random(917449)
        draws = sorted(
            sum(sum(rng.choices(g, k=len(g))) / len(g) for g in groups) / len(groups)
            for _ in range(2000)
        )
        scheduled[key] = [draws[49], draws[1949]]
    return {
        "binary_approximate95": result,
        "scheduled_delta_stratified_bootstrap95": scheduled,
        "method": "Equal mixture of fixed templates. Binary bounds average Bonferroni-adjusted stratum "
        "score limits. The stratified episode bootstrap quantifies empirical cost variability; "
        "when all sampled changes equal zero, its collapsed interval does not exclude rare effects.",
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--root", required=True, type=Path)
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    plan, pid = load_plan(args.root / "preregistration.json")
    rows = []
    strata = {}
    for s in plan["cohorts"]:
        group = reconcile(args.root, plan, pid, s)
        strata[s["split"]] = summarize(group)
        rows.extend(group)
    summary = summarize(rows)
    gain = [
        r["mean_paired_delta"]["utility"] for r in rows if r["mean_paired_delta"]["utility"] is not None
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "episode-index.json", rows)
    save(
        args.out / "summary.json",
        {
            "plan_id": pid,
            "bank_id": plan["bank_id"],
            "strata": strata,
            "pooled": summary,
            "fixed_mixture_uncertainty": design_intervals(rows, strata),
            "unit": "32 distinct seeded instances in two equally weighted fixed templates, K=1. "
            "Wilson intervals describe binomial sampling uncertainty; bootstrap intervals resample episodes, "
            "not templates or repeated branches. No estimate of cross-template uncertainty.",
            "precision_planning": {
                "n_independent_episodes_with_contrast": len(gain),
                "observed_utility_sd": statistics.stdev(gain) if len(gain) >= 2 else None,
                "illustrative_tasks_for_95pct_chance_of_at_least_one_opportunity": {
                    str(p): math.ceil(math.log(0.05) / math.log1p(-p)) for p in (0.01, 0.02, 0.05)
                },
                "scope": "Prospective independent-task Bernoulli scenarios. Assumed opportunity rates are not pilot estimates. "
                "Formal effect-power sizing also requires nonzero support and independent contrasts.",
            },
            "precision": "Small event counts and sparse matched tasks limit formal sample planning. "
            "A null conditional effect is unavailable when no pairs complete. A zero observed event count "
            "among observed pairs has a nonzero interval upper bound.",
            "interpretation": "Full-population bounds are descriptive missing-outcome bounds for these 32 tasks. "
            "The scheduled delta includes structural no-op tasks and measured matching overhead versus bare; "
            "it does not impute a conditional paired effect when there are no pairs.",
            "cost_scope": "Offline shadow matching was timed after archive load. Bare actor and paired "
            "generation and tool costs use the archived phase-specific accounting scope. Prior common tool "
            "timing and USD are unknown; branch wall time includes verification.",
        },
    )
    print(args.out / "summary.json")


if __name__ == "__main__":
    main()
