"""Mini-data source evidence, checkpoint restoration and target intervention runs."""

import argparse
import json
import math
import random
import uuid
from pathlib import Path

from . import mini
from .branches import branch, load_actor, revision_fits
from .models import FrozenQwen, xml_parser
from .records import digest, read, save
from .replay import load_run, nodes, protocol_digest, replay_tape, require, restore, selected_turn


# A reviewed mapping from the exact frozen teacher proposal to a mini-DSL
# applicability predicate. New free-text rules require their own explicit mapping.
CONTRACT = "all-records-ascending-after-ASC/v1"


def reviewed_contract(candidate):
    registry = read(Path(__file__).with_name("reviewed_contracts.json"))
    require(registry["schema"] == 1, "unknown review registry version")
    return registry["contracts_by_candidate_digest"].get(digest(candidate))


def proposal_source(run, item):
    steps = []
    for turn, folder, _ in nodes(run, item):
        draft, execution = read(folder / "draft.json"), read(folder / "execution.json")
        steps.append(
            {
                "step": turn,
                "a0": draft["a0"],
                "error": draft["error"],
                "executed": execution["executed_action"],
                "observation": execution["observation"],
            }
        )
    return {
        "run_id": item["run_id"],
        "episode_id": item["episode_id"],
        "task": item["task"],
        "steps": steps,
        "score": item["verifier"],
    }


def source_verify(args):
    manifest, rows = load_run(args.run)
    require(manifest["split"] == "source", "source run required")
    args.out.mkdir(parents=True, exist_ok=False)
    evidence, cards = [], []
    for path in sorted((args.run / "proposals").glob("*.json")):
        p = read(path)
        result = {
            "proposal": path.name,
            "proposal_digest": digest(p),
            "episode_id": p.get("episode_id"),
            "teacher_usage": p.get("usage"),
            "teacher_seconds": p.get("seconds"),
        }
        try:
            require(
                p["status"] == "ok"
                and p["model"] == p["response_model"] == "gpt-6-sol"
                and p["run_id"] == manifest["run_id"],
                "invalid teacher/source provenance",
            )
            require(
                not manifest["actor"]["do_sample"] and manifest["actor"]["temperature"] == 0,
                "stochastic source needs fresh held-out continuations",
            )
            item = next((r for r in rows if r["episode_id"] == p["episode_id"]), None)
            require(
                item is not None and item["verifier"]["status"] == "fail", "not a failed source episode"
            )
            require(
                digest(proposal_source(args.run, item)) == p["source_digest"],
                "source trace digest mismatch",
            )
            c = p["candidate"]
            turn = c["step"]
            require(
                type(turn) is int
                and 1 <= turn <= len(item["steps"])
                and item["steps"][turn - 1]["executable"],
                "proposal has no executable source anchor",
            )
            require(
                c["alternative"]
                and all(
                    op in {"SHOW", "ASC", "DESC", "UNIQUE", "POSITIVE", "SUBMIT"}
                    for op in c["alternative"]
                ),
                "unsupported repair operation",
            )
            repair = {
                "commands": [{"keystrokes": op, "duration": 0.0} for op in c["alternative"]],
                "complete": False,
            }
            result.update(turn=turn, replacement=repair, confirmations=[])
            for phase, order in [
                ("discovery", ("original", "repair")),
                ("confirmation", ("repair", "original")),
            ]:
                scores = {}
                for side in order:
                    trace = replay_tape(
                        args.run,
                        manifest,
                        item,
                        start=turn,
                        replacement=repair if side == "repair" else None,
                    )
                    ref = f"replays/{digest(p)}/{phase}-{side}.json"
                    save(args.out / ref, trace)
                    scores[side] = {
                        "evidence": ref,
                        "digest": digest(trace),
                        "verdict": trace["score"],
                        "termination": trace["termination"],
                        "submitted": trace["final_state"]["submitted"],
                    }
                result["confirmations"].append(
                    {
                        "phase": phase,
                        "order": order,
                        "arms": scores,
                        "delta_success": scores["repair"]["verdict"]["reward"]
                        - scores["original"]["verdict"]["reward"],
                    }
                )
                require(
                    scores["original"]["verdict"]["status"] == "fail"
                    and scores["repair"]["verdict"]["status"] == "pass",
                    "source contrast not reproduced",
                )
            result["source_verified"] = True
            require(
                reviewed_contract(c) == CONTRACT,
                "validated repair, no reviewed executable applicability rule",
            )
            pre, draft, _ = restore(args.run, item, turn)
            require(
                item["task"]["family"] == "ascending"
                and draft["a0"]["commands"]
                and applies(pre, item["task"]) == 1,
                "unsupported source state",
            )
            card = {
                "contract": CONTRACT,
                "source_run": manifest["run_id"],
                "source_episode": item["episode_id"],
                "checkpoint_id": draft["checkpoint_id"],
                "a0": draft["a0"],
                "candidate": c,
                "evidence_digest": digest(result["confirmations"]),
                "teacher_proposal_digest": digest(p),
                "scope": "deterministic same-source-state composite repair; no unique-cause attribution",
            }
            card["memory_id"] = digest(card)
            cards.append(card)
            result.update(status="accepted", memory_id=card["memory_id"])
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result.update(status="excluded", reason=str(exc))
        evidence.append(result)
    save(args.out / "source-evidence.json", evidence)
    bank = {
        "schema": 1,
        "protocol": protocol_digest(),
        "source_manifest_digest": digest(manifest),
        "source_inputs": sorted({digest(item["task"]["input"]) for item in rows}),
        "evidence_index_digest": digest(evidence),
        "cards": cards,
    }
    save(args.out / "bank.json", {**bank, "bank_id": digest(bank)})
    print(f"accepted {len(cards)} / {len(evidence)} proposals")


def archive_digest(run, item):
    records = []
    for _, folder, _ in nodes(run, item):
        records.append({p.name: digest(read(p)) for p in sorted(folder.glob("*.json"))})
    return digest(
        [
            item,
            records,
            read(run / item["episode_id"] / "sampling.json"),
            read(run / item["episode_id"] / "verdict.json"),
        ]
    )


def audit_aa(args):
    manifest, rows = load_run(args.run)
    args.out.mkdir(parents=True, exist_ok=False)
    actor = load_actor(manifest) if args.free else None
    checks = []
    for item in rows:
        eid = item["episode_id"]
        record = {"episode_id": eid, "free_status": "not_requested"}
        try:
            turn = selected_turn(args.run, manifest, item)
            # Cover the selected checkpoint; also verify histories of tasks which
            # end before their scheduled opportunity.
            anchor = turn or 1
            record.update(
                selected_turn=turn, tape_anchor=anchor, archive_digest=archive_digest(args.run, item)
            )
            traces = []
            for arm in ("a", "b"):
                trace = replay_tape(args.run, manifest, item, start=anchor)
                save(args.out / eid / f"tape-{arm}.json", trace)
                traces.append(trace)
            require(
                traces[0]["trace_digest"] == traces[1]["trace_digest"]
                and traces[0]["score"]["reward"] == traces[1]["score"]["reward"],
                "A/A replay discrepancy",
            )
            record.update(tape_status="pass", trace_digest=traces[0]["trace_digest"])
            if actor:
                # Even if the scheduled point is unreached, measure model drift
                # on an observed generation. A forced fixture prefix is never regenerated.
                i = turn or (2 if manifest.get("cohort") == "controlled_fixed_prefix" else 1)
                signatures = []
                for arm in ("a", "b"):
                    live = branch(
                        args.run,
                        manifest,
                        item,
                        i,
                        actor,
                        "free",
                        "",
                        args.seed,
                        args.total_tokens,
                        args.total_tools,
                    )
                    save(args.out / eid / f"free-{arm}.json", live)
                    require(
                        live["score"]["reward"] is not None and live["requests"], "unavailable free A/A"
                    )
                    signatures.append(
                        digest(
                            [
                                [r["generation"]["text"] for r in live["requests"]],
                                live["events"],
                                live["score"]["reward"],
                            ]
                        )
                    )
                record.update(
                    free_status="agree" if signatures[0] == signatures[1] else "drift",
                    free_anchor=i,
                    free_digests=signatures,
                )
        except (ValueError, KeyError, TypeError, OSError) as exc:
            if record.get("tape_status") == "pass":
                record.update(free_status="unavailable", free_error=str(exc))
            else:
                record.update(tape_status="failed", reason=str(exc))
        checks.append(record)
    report = {
        "schema": 1,
        "run_id": manifest["run_id"],
        "manifest_digest": digest(manifest),
        "protocol": protocol_digest(),
        "checks": checks,
    }
    save(args.out / "aa.json", {**report, "audit_id": digest(report)})
    print([(r["episode_id"], r["tape_status"], r["free_status"]) for r in checks])


def applies(pre, spec):
    if spec["family"] != "ascending":
        return 0
    state = pre["state"]
    if state["submitted"] is not None or state["working"] != sorted(state["input"]):
        return 0
    # The observable prior tool result must support the stated ASC prerequisite.
    return int(
        any(
            m["role"] == "user"
            and any(line.startswith("ascending:") for line in m["content"].splitlines())
            for m in pre["messages"][2:]
        )
    )


def match(cards, spec, pre, draft):
    if spec["family"] != "ascending" or not cards:
        return {"status": "no_candidate", "memory_id": None, "v": None, "rho": None, "e": None}
    card = sorted(cards, key=lambda v: v["memory_id"])[0]
    rho = applies(pre, spec)
    ops = [c["keystrokes"].strip() for c in draft["a0"]["commands"]]
    # Only the observed source action signature and a directly protective
    # submit are supported by this first, intentionally narrow exposure rule.
    if ops == ["SUBMIT"]:
        exposure = 0
    elif ops == ["POSITIVE"]:
        exposure = 1
    else:
        exposure = None
    status = (
        "inapplicable" if not rho else {0: "avoided", 1: "exposed", None: "unknown_exposure"}[exposure]
    )
    return {
        "status": status,
        "memory_id": card["memory_id"],
        "v": 1,
        "rho": rho,
        "e": exposure,
        "rule": CONTRACT,
        "observed_ops": ops,
        "reminder": card["candidate"]["reminder"],
    }


def open_bank(path):
    bank = read(path)
    bank_id = bank.pop("bank_id")
    require(
        bank_id == digest(bank) and bank["protocol"] == protocol_digest(),
        "changed/incompatible memory bank",
    )
    evidence = read(path.parent / "source-evidence.json")
    require(digest(evidence) == bank["evidence_index_digest"], "source evidence index changed")
    for card in bank["cards"]:
        require(
            card["contract"] == CONTRACT
            and reviewed_contract(card["candidate"]) == CONTRACT
            and digest({k: v for k, v in card.items() if k != "memory_id"}) == card["memory_id"],
            "unsupported card",
        )
        result = next(
            (
                r
                for r in evidence
                if r.get("memory_id") == card["memory_id"] and r["status"] == "accepted"
            ),
            None,
        )
        require(
            result and digest(result["confirmations"]) == card["evidence_digest"],
            "missing source evidence",
        )
        for phase in result["confirmations"]:
            for side in phase["arms"].values():
                require(
                    digest(read(path.parent / side["evidence"])) == side["digest"],
                    "changed source replay",
                )
    return bank_id, bank["cards"], bank["source_inputs"]


def pair(args):
    manifest, rows = load_run(args.run)
    require(
        manifest["split"] in {"target_train", "target_val", "target_test", "controlled_transfer"},
        "a target or separately declared control run is required",
    )
    bank_id, cards, source_inputs = open_bank(args.bank)
    require(all(digest(t["input"]) not in source_inputs for t in manifest["tasks"]),
            "target input duplicates an input in the frozen source run")
    aa = read(args.aa)
    audit_id = aa.pop("audit_id")
    require(
        digest(aa) == audit_id
        and aa["manifest_digest"] == digest(manifest)
        and aa["run_id"] == manifest["run_id"]
        and aa["protocol"] == protocol_digest(),
        "A/A audit is for another run or protocol",
    )
    require(args.replicates >= 1 and args.total_tokens > 0 and args.total_tools > 0, "invalid budget")
    require(
        all(
            math.isfinite(v) and v >= 0 for v in (args.lambda_tokens, args.lambda_time, args.lambda_tool)
        ),
        "utility weights must be finite and nonnegative",
    )
    args.out.mkdir(parents=True, exist_ok=False)
    policy = {
        "schema": 1,
        "protocol": protocol_digest(),
        "bank_id": bank_id,
        "audit_id": audit_id,
        "run_id": manifest["run_id"],
        "split": manifest["split"],
        "cohort": manifest.get("cohort", "natural"),
        "replicates": args.replicates,
        "seed": args.seed,
        "coupling": "shared relative suffix seed; separate revision seed",
        "limits": {
            "actor_calls": manifest["max_steps"],
            "steps": manifest["max_steps"],
            "total_tokens": args.total_tokens,
            "tool_calls": args.total_tools,
        },
        "comparison": "complete two-arm pair; no treatment propensity",
        "neutral": "not collected; observed effect combines reminder content and resampling",
        "utility_weights": {
            "per_1000_tokens": args.lambda_tokens,
            "per_tool": args.lambda_tool,
            "per_minute": args.lambda_time,
        },
        "time_scope": "common archived actor generation + new actor generation + new mini tool execution. "
        "Prior tool times unrecorded and common to both arms. Excludes restore, tokenization, "
        "ledger I/O, verification and queue time. Raw branch wall time is separate.",
        "selection": "one scheduled opportunity; one card; ascending prerequisite; unknown exposure retained",
    }
    save(args.out / "protocol.json", policy)
    # Freeze the entire match ledger before consulting A/A results or generating
    # any branch. Only the eligible prefix and original proposal enter match().
    funnel = []
    for item in rows:
        turn = selected_turn(args.run, manifest, item)
        decision = {
            "episode_id": item["episode_id"],
            "q": item["q"],
            "main_opportunity": item["main_opportunity"],
            "turn": turn,
            "status": "never_reached",
            "memory_id": None,
        }
        if turn is not None:
            pre, draft, _ = restore(args.run, item, turn)
            decision.update(
                match(cards, item["task"], pre, draft),
                checkpoint_id=pre["checkpoint_id"],
                a0_digest=draft["a0_digest"],
                pre_digest=digest(pre),
            )
        funnel.append(decision)
    save(args.out / "matches.json", funnel)
    actor = None
    pairs = []
    for item, decision in zip(rows, funnel, strict=True):
        if decision.get("rho") != 1:
            continue
        eid, turn = item["episode_id"], decision["turn"]
        audit = next((r for r in aa["checks"] if r["episode_id"] == eid), {})
        if (
            audit.get("tape_status") != "pass"
            or audit.get("free_status") != "agree"
            or audit.get("selected_turn") != turn
            or audit.get("archive_digest") != archive_digest(args.run, item)
        ):
            pairs.append(
                {"episode_id": eid, "status": "restore_unapproved", "route": "prospective_single_arm"}
            )
            continue
        if actor is None:
            actor = load_actor(manifest)
        pre, draft, _ = restore(args.run, item, turn)
        if not revision_fits(
            manifest, item, turn, actor, pre, draft, decision["reminder"], args.total_tokens
        ):
            pairs.append({"episode_id": eid, "status": "insufficient_revision_budget"})
            continue
        for rep in range(args.replicates):
            pair_id = digest([digest(policy), decision, rep])
            seed = random.Random(f"coupling:{args.seed}:{eid}:{rep}").randrange(2**31)
            order = ["silent", "remind"]
            random.Random("order:" + pair_id).shuffle(order)
            result = {
                "pair_id": pair_id,
                "episode_id": eid,
                "checkpoint_id": pre["checkpoint_id"],
                "a0_digest": draft["a0_digest"],
                "memory_id": decision["memory_id"],
                "replicate": rep,
                "seed": seed,
                "order": order,
                "status": "incomplete",
                "arms": {},
            }
            for arm in order:
                live = branch(
                    args.run,
                    manifest,
                    item,
                    turn,
                    actor,
                    arm,
                    decision["reminder"],
                    seed,
                    args.total_tokens,
                    args.total_tools,
                )
                ref = f"pairs/{pair_id}/{arm}.json"
                save(args.out / ref, live)
                u = live["usage"]
                seconds = u["common_actor_seconds"] + u["new_actor_seconds"] + u["new_tool_seconds"]
                cost = (
                    args.lambda_tokens * (u["input_tokens"] + u["output_tokens"]) / 1000
                    + args.lambda_tool * u["tool_calls"]
                    + args.lambda_time * seconds / 60
                )
                y = live["score"]["reward"]
                result["arms"][arm] = {
                    "file": ref,
                    "digest": digest(live),
                    "Y": y,
                    "utility": y - cost if y is not None else None,
                    "cost_penalty": cost,
                    "accounted_seconds": seconds,
                    "revision_calls": live["revision_calls"],
                    "usage": u,
                }
            a, b = result["arms"]["silent"], result["arms"]["remind"]
            if a["Y"] is not None and b["Y"] is not None:
                result.update(
                    status="complete",
                    difference=b["utility"] - a["utility"],
                    regression=int(a["Y"] == 1 and b["Y"] == 0),
                    rescue=int(a["Y"] == 0 and b["Y"] == 1),
                )
            pairs.append(result)
            save(args.out / "pairs.json", pairs)
    save(args.out / "pairs.json", pairs)
    print(
        f"{len(funnel)} scheduled episodes; {sum(p['status'] == 'complete' for p in pairs)} complete pairs"
    )


def controlled(args):
    from .__main__ import collect_episode

    require(
        args.episodes >= 1 and args.max_steps >= 3, "controlled prefix requires at least three turns"
    )
    actor = FrozenQwen(args.model, args.max_new_tokens, 0.0)

    class FixedPrefix:
        def __init__(self):
            self.first = True

        def generate(self, messages, seed):
            if not self.first:
                return actor.generate(messages, seed)
            self.first = False
            text = (
                "<response><analysis></analysis><plan></plan><commands>"
                '<keystrokes duration="0">ASC</keystrokes></commands></response>'
            )
            return {
                "text": text,
                "status": "ok",
                "seed": seed,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "seconds": 0.0,
                "origin": "scripted_control_prefix",
                "prompt_digest": digest(actor.prompt(messages)),
            }

    args.out.mkdir(parents=True, exist_ok=False)
    tasks = [mini.task("controlled_transfer", args.seed + i) for i in range(args.episodes)]
    manifest = {
        "schema": 1,
        "run_id": uuid.uuid4().hex,
        "benchmark": "mini-data-v1",
        "cohort": "controlled_fixed_prefix",
        "split": "controlled_transfer",
        "seed": args.seed,
        "tasks": tasks,
        "actor": actor.info,
        "max_steps": args.max_steps,
        "memory_enabled": False,
        "teacher_enabled": False,
        "protocol": "fixed first ASC executed by mini; second opportunity predetermined, natural Qwen draft",
        "parser": "TerminusXMLPlainParser",
        "split_plan": mini.TEMPLATES,
    }
    save(args.out / "manifest.json", manifest)
    parser = xml_parser()
    with (args.out / "index.jsonl").open("x", encoding="utf-8") as f:
        for i, spec in enumerate(tasks):
            item = collect_episode(
                args.out,
                FixedPrefix(),
                parser,
                spec,
                f"ep-{i:04d}",
                args.seed + i,
                args.max_steps,
                fixed_opportunity=2,
            )
            item.update(run_id=manifest["run_id"], split=manifest["split"])
            f.write(json.dumps(item) + "\n")
            f.flush()


def main():
    cli = argparse.ArgumentParser()
    group = cli.add_subparsers(dest="mode", required=True)
    for name in ("source-verify", "aa", "pair", "prepare-control"):
        p = group.add_parser(name)
        p.add_argument("--out", required=True, type=Path)
        if name != "prepare-control":
            p.add_argument("--run", required=True, type=Path)
        if name in ("aa", "pair"):
            p.add_argument("--seed", type=int, default=7331)
            p.add_argument("--total-tokens", type=int, default=16384)
            p.add_argument("--total-tools", type=int, default=64)
        if name == "aa":
            p.add_argument("--free", action="store_true")
        if name == "pair":
            p.add_argument("--bank", required=True, type=Path)
            p.add_argument("--aa", required=True, type=Path)
            p.add_argument("--replicates", type=int, default=1)
            p.add_argument("--lambda-tokens", type=float, default=0.01)
            p.add_argument("--lambda-tool", type=float, default=0.001)
            p.add_argument("--lambda-time", type=float, default=0.01)
        if name == "prepare-control":
            p.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-7B")
            p.add_argument("--episodes", type=int, default=2)
            p.add_argument("--seed", type=int, default=6200)
            p.add_argument("--max-steps", type=int, default=4)
            p.add_argument("--max-new-tokens", type=int, default=256)
    args = cli.parse_args()
    {"source-verify": source_verify, "aa": audit_aa, "pair": pair, "prepare-control": controlled}[
        args.mode
    ](args)


if __name__ == "__main__":
    main()
