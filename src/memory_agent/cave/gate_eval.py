"""Evaluate the frozen threshold on validation archives, with shared matching and timed CPU decisions."""

import argparse
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import formal, formal_plan as fp, gate
from .records import digest, read, save
from .replay import require, restore


POLICIES = (
    "always_silent",
    "always_remind",
    "ready_submit_rule",
    "exposure_rule",
    "learned",
    "learned_no_a0",
)
ROUNDS = 101


def decide(name, model, x):
    if name == "always_silent":
        gate.predict(model, x)
        return False
    if name == "always_remind":
        return True
    if name == "ready_submit_rule":
        return not gate.ready(x, model["cards"])
    if name == "exposure_rule":
        e = x["evidence"]
        return e["source_verified"] == e["applicable"] == e["exposed"] == 1
    return gate.predict(model, x, "no_a0" if name == "learned_no_a0" else "full")[
        "remind"
    ]


def benchmark(model, row, pre, draft, decision, plan):
    totals = Counter()
    decisions = {}
    for k in range(ROUNDS + 1):
        order = list(POLICIES)
        random.Random(f"gate-timing:{row['seed']}:{k}").shuffle(order)
        for name in order:
            start = time.perf_counter_ns()
            x = formal.gate_inputs(row, pre, draft, decision, plan)
            action = decide(name, model, x)
            elapsed = time.perf_counter_ns() - start
            if k:
                require(action == decisions[name], "nondeterministic policy")
                totals[name] += elapsed
            else:
                decisions[name] = action
    return decisions, {name: n / ROUNDS / 1e9 for name, n in totals.items()}


def keyed(rows, key):
    index = {r[key]: r for r in rows}
    require(len(index) == len(rows), "duplicate validation entry")
    return index


def load_validation(root, model):
    plan, pid = fp.load(root)
    mf, rows = fp.checked_run(plan, "validation", root / "validation")
    release = gate.signed(root / "dataset-v1/dataset.json", "dataset_id")
    require(
        release["bank_id"] == model["bank_id"] == plan["bank_id"]
        and mf["actor"] == plan["actor"]
        and digest(plan["actor"]) == model["actor_digest"]
        and release["plan_id"] == pid
        and plan["protocol"] == model["engine"]
        and release["test_manifest_digest"] == plan["test_manifest_digest"],
        "model and validation release lineage differ",
    )
    if release["schema"] == "frozen-gate-validation/v1":
        lock = read(root / "model-before-pairing.json")
        require(
            lock["model_digest"] == digest(model) == release["model_before_pairing"]
            and lock["plan_id"] == pid
            and lock["blind_digest"]
            == digest(read(root / "validation/blind-seal.json")),
            "prospective model lock mismatch",
        )
    else:
        require(
            release["schema"] == "formal-paired-dataset/v1"
            and release["splits"]["validation"]["audit_seal"]
            == formal.check_audit(root, "validation", pid, mf["run_id"]),
            "formal validation seal mismatch",
        )
    samples, evidence, ledger = (
        gate.lines(root / "dataset-v1" / (f"validation{s}.jsonl"))
        for s in ("", "-evidence", "-episodes")
    )
    info = release["splits"]["validation"]
    require(
        [digest(x) for x in (samples, evidence, ledger)]
        == [info[k + "_digest"] for k in ("samples", "evidence", "ledger")]
        and len(rows) == len(ledger) == info["episodes"]
        and len(samples) == len(evidence) == info["samples"],
        "validation export changed",
    )
    require(
        not set(model["task_ids"]).intersection(digest(r["task"]) for r in rows),
        "train and validation task overlap",
    )
    shadow = read(root / "audit-validation/shadow.json")
    require(
        shadow["plan_id"] == pid and shadow["run_id"] == mf["run_id"],
        "retrieval timing provenance mismatch",
    )
    matches = read(root / "audit-validation/pairs/matches.json")
    return (
        plan,
        release,
        rows,
        ledger,
        keyed(samples, "sample_id"),
        keyed(evidence, "sample_id"),
        matches,
        shadow["rows"],
    )


def score(raw, policy_seconds, plan):
    if not raw:
        return None
    metrics = {
        k: raw[k]
        for k in (
            "success",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "actor_calls",
            "tool_calls",
            "accounted_seconds",
        )
    }
    sec = metrics["accounted_seconds"]
    metrics["accounted_seconds"] = None if sec is None else sec + policy_seconds
    if any(
        metrics[k] is None
        for k in ("success", "total_tokens", "tool_calls", "accounted_seconds")
    ):
        metrics["utility"] = None
    else:
        metrics["utility"] = (
            metrics["success"]
            - plan["lambda_tokens"] * metrics["total_tokens"] / 1000
            - plan["lambda_tool"] * metrics["tool_calls"]
            - plan["lambda_time"] * metrics["accounted_seconds"] / 60
        )
    return metrics


def evaluate_rows(root, model):
    plan, release, rows, ledger, samples, evidence, matches, shadows = load_validation(
        root, model
    )
    out = []
    used = set()
    for row, entry, d, sh in zip(rows, ledger, matches, shadows, strict=True):
        eid = row["episode_id"]
        require(
            all(t["episode_id"] == eid for t in (entry, d, sh))
            and entry["run_id"] == sh["run_id"] == row["run_id"]
            and digest(row["task"]) == entry["task_digest"]
            and entry["decision"] == d["status"] == sh["status"]
            and row["main_turn_id"] == d["turn"]
            and d["q"] == row["q"] == entry["q"],
            "validation episode alignment changed",
        )
        inp, output = sh["bare_input_tokens"], sh["bare_output_tokens"]
        bare = {
            "success": row["verifier"]["reward"],
            "input_tokens": inp,
            "output_tokens": output,
            "total_tokens": None if inp is None or output is None else inp + output,
            "actor_calls": sh["bare_actor_calls"],
            "tool_calls": sh["bare_tool_calls"],
            "accounted_seconds": sh["bare_actor_seconds"],
        }
        require(
            read(root / "validation" / eid / "verdict.json") == row["verifier"],
            "changed bare verdict",
        )
        funded = (
            d.get("rho") == 1
            and d["turn"] < plan["max_steps"]
            and "insufficient_revision_budget" not in entry["exclusion"]
        )
        arms = None
        sid = entry["sample_id"]
        preds = {}
        acts = dict.fromkeys(POLICIES, False)
        times = dict.fromkeys(POLICIES, 0.0)
        if funded:
            pre, draft, _ = restore(root / "validation", row, d["turn"])
            require(
                digest(pre) == d["pre_digest"] and draft["a0_digest"] == d["a0_digest"],
                "anchor drift",
            )
            x = formal.gate_inputs(row, pre, draft, d, plan)
            acts, times = benchmark(model, row, pre, draft, d, plan)
            preds = {v: gate.predict(model, x, v) for v in ("full", "no_a0")}
            if sid:
                used.add(sid)
                sm, ev = samples[sid], evidence[sid]
                require(
                    sm["features"] == x
                    and ev["checkpoint"] == "validation/" + d["checkpoint_id"]
                    and ev["decision_digest"] == digest(d)
                    and ev["plan_id"] == release["plan_id"]
                    and ev["pair_id"] in entry["pair_ids"]
                    and ev["task_digest"] == entry["task_digest"],
                    "exported checkpoint features or pair provenance differ",
                )
                arms = {}
                for side in ("silent", "remind"):
                    proof = ev["arms"][side]
                    expected = (
                        f"audit-validation/pairs/pairs/{ev['pair_id']}/{side}.json"
                    )
                    require(proof["path"] == expected, "arm link mismatch")
                    live = read(root / expected)
                    a = proof["observed"]
                    require(
                        digest(live) == proof["digest"]
                        and live["arm"] == side
                        and live["score"]["reward"] == a["success"]
                        and all(a[k] == v for k, v in live["usage"].items())
                        and a["total_tokens"] == a["input_tokens"] + a["output_tokens"]
                        and math.isclose(
                            a["accounted_seconds"],
                            sum(
                                a[k]
                                for k in (
                                    "common_actor_seconds",
                                    "new_actor_seconds",
                                    "new_tool_seconds",
                                )
                            ),
                            abs_tol=1e-10,
                        )
                        and math.isclose(
                            score(a, 0, plan)["utility"], a["utility"], abs_tol=1e-10
                        ),
                        "archived arm cost or outcome drift",
                    )
                    arms[side] = a
                require(
                    math.isclose(
                        sm["targets"]["D"],
                        arms["remind"]["utility"] - arms["silent"]["utility"],
                        abs_tol=1e-9,
                    )
                    and sm["targets"]["H"]
                    == int(
                        arms["silent"]["success"] == 1
                        and arms["remind"]["success"] == 0
                    ),
                    "paired targets drift",
                )
        policies = {
            "bare_actor": {
                "action": False,
                "gate_seconds": 0,
                "retrieval_seconds": 0,
                "metrics": score(bare, 0, plan),
                "harm": 0,
                "rescue": 0,
            }
        }
        for name in POLICIES:
            act = acts[name]
            raw = (
                arms["remind" if act else "silent"] if arms else (None if act else bare)
            )
            ss = arms["silent"]["success"] if arms else None
            rr = arms["remind"]["success"] if arms else None
            policies[name] = {
                "action": act,
                "gate_seconds": times[name],
                "retrieval_seconds": sh["match_seconds"],
                "metrics": score(raw, times[name] + sh["match_seconds"], plan),
                "harm": int(ss == 1 and rr == 0)
                if act and arms
                else (None if act else 0),
                "rescue": int(ss == 0 and rr == 1)
                if act and arms
                else (None if act else 0),
            }
        out.append(
            {
                "episode_id": eid,
                "task_id": entry["task_digest"],
                "template_id": entry["template_id"],
                "scheduled_opportunity": entry["scheduled_opportunity"],
                "reached": row["main_reached"],
                "applicable": d.get("rho") == 1,
                "funded_or_no_known_budget_failure": funded,
                "exclusions": entry["exclusion"],
                "sample_id": sid,
                "actual_D": samples[sid]["targets"]["D"] if sid else None,
                "actual_H": samples[sid]["targets"]["H"] if sid else None,
                "predictions": preds,
                "policies": policies,
            }
        )
    require(
        used == set(samples) == set(evidence),
        "validation pairs omitted or wrongly attached",
    )
    return release, out


def average(values):
    observed = [v for v in values if v is not None]
    return {
        "mean": statistics.fmean(observed) if observed else None,
        "observed": len(observed),
        "total": len(values),
    }


def differences(rows, policy, ref, metric):
    vals = []
    for row in rows:
        a, b = (row["policies"][n]["metrics"] for n in (policy, ref))
        vals.append(
            a[metric] - b[metric]
            if a and b and a[metric] is not None and b[metric] is not None
            else None
        )
    return vals


def stratified_ci(rows, values):
    if not rows or any(v is None for v in values):
        return None
    groups = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        groups[row["template_id"]].append(value)
    rng = random.Random(9009)
    means = sorted(
        sum(sum(rng.choices(a, k=len(a))) for a in groups.values()) / len(rows)
        for _ in range(1000)
    )
    return [means[24], means[974]]


def summaries(rows):
    results = {}
    for name in ("bare_actor", *POLICIES):
        p = [r["policies"][name] for r in rows]
        summary = {
            "reminders": sum(v["action"] for v in p),
            "episodes": len(rows),
            "harm": average([v["harm"] for v in p]),
            "rescue": average([v["rescue"] for v in p]),
            "gate_seconds": average([v["gate_seconds"] for v in p]),
            "retrieval_seconds": average([v["retrieval_seconds"] for v in p]),
        }
        if rows and all(v["harm"] == 0 for v in p):
            summary["zero_harm_one_sided_95_upper"] = 1 - 0.05 ** (1 / len(rows))
        for metric in (
            "success",
            "utility",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "actor_calls",
            "tool_calls",
            "accounted_seconds",
        ):
            summary[metric] = average(
                [v["metrics"][metric] if v["metrics"] else None for v in p]
            )
        for ref in ("always_silent", "always_remind", "bare_actor"):
            deltas = {}
            for metric in (
                "success",
                "utility",
                "total_tokens",
                "tool_calls",
                "accounted_seconds",
            ):
                values = differences(rows, name, ref, metric)
                deltas[metric] = average(values)
                if metric == "utility":
                    deltas[metric]["stratified_task_bootstrap_95"] = stratified_ci(
                        rows, values
                    )
            summary["delta_vs_" + ref] = deltas
        summary["reminder_rate"] = summary["reminders"] / len(rows) if rows else None
        summary["rescue_count"] = sum(v["rescue"] == 1 for v in p)
        summary["regression_count"] = sum(v["harm"] == 1 for v in p)
        summary["unscored_interventions"] = sum(
            v["action"] and v["harm"] is None for v in p
        )
        results[name] = summary
    return results


def evaluate(args):
    model = gate.load_model(args.model)
    release, rows = evaluate_rows(args.root, model)
    complete = [r for r in rows if r["sample_id"]]
    diagnostics = {}
    for v in ("full", "no_a0"):
        errors = [r["predictions"][v]["predicted_D"] - r["actual_D"] for r in complete]
        diagnostics[v] = {
            "pairs": len(complete),
            "D_mae": average([abs(e) for e in errors]),
            "D_rmse": math.sqrt(statistics.fmean(e * e for e in errors))
            if errors
            else None,
            "observed_harms": sum(r["actual_H"] for r in complete),
            "risk_status": model["risk_status"],
        }
    summary = {
        "schema": "frozen-gate-comparison/v1",
        "model_id": model["model_id"],
        "validation_dataset_id": release["dataset_id"],
        "plan_id": release["plan_id"],
        "evaluation_code": fp.code_digest("gate_eval.py"),
        "threshold": model["threshold"],
        "design_budgets": {
            "mean_extra_tokens_per_task": model["label_spec"][
                "policy_mean_extra_tokens_budget"
            ],
            "mean_regression_rate": model["label_spec"]["policy_risk_delta"],
        },
        "threshold_status": "Preregistered training-only cutoff; validation is diagnostic and does not change parameters.",
        "diagnostics": diagnostics,
        "funnel": {
            "episodes": len(rows),
            "reached": sum(r["reached"] for r in rows),
            "applicable": sum(r["applicable"] for r in rows),
            "complete_pairs": len(complete),
        },
        "all_tasks": summaries(rows),
        "complete_pairs": summaries(complete),
        "episode_ledger_digest": digest(rows),
        "scope": "Exact offline policy replay at most once per task where both recorded arms exist. A missing intervention is unknown. Always-remind is limited to applicable, budget-eligible candidates, identical to the applicability-only rule on this archive.",
        "timing": "Archived prefix actor and branch suffix actor/tool time plus archived in-memory matching and interleaved CPU policy timing, 101 repetitions after warmup. No additional model calls, tokens or tools for the gate. Baseline tool wall time and pre-fork tool wall time were not logged. Offline artifact I/O and verifier are excluded. Always-silent runs the full gate before ignoring its output.",
        "uncertainty": "Task bootstrap within template at fixed observed mix. Zero-event upper bound assumes independent Bernoulli tasks and covers each policy separately. Conditional pair results describe completed opportunities. No calibrated risk head when training harms are absent.",
    }
    args.out.mkdir(parents=True, exist_ok=False)
    formal.write_rows(args.out / "policy-episodes.jsonl", rows)
    save(args.out / "comparison.json", summary)
    print(args.out / "comparison.json")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    evaluate(cli.parse_args())
