"""Export a reviewed development-only descending card from an independent source package."""

import argparse
from collections import Counter
from pathlib import Path

from . import mini
from .experiment import proposal_source
from .records import digest, read, save, verify
from .replay import load_run, protocol_digest, require, restore


CANDIDATE = "5df3b2ec4f8f23bb80536e7905af514a5f7a23c85b648f95ad14191b765d2a3a"
SOURCE_INSTRUCTION = "Order the entire sequence from greatest to least, retaining repeated integers, and submit the result."


def applicable(task, state):
    """Operating domain: explicit full-list descending mini tasks with intact records."""
    return (
        task["family"] == "descending"
        and task["instruction"] in {SOURCE_INSTRUCTION, mini.TEMPLATES["target_train"][1]}
        and state["submitted"] is None
        and state["input"] == task["input"]
        and Counter(state["working"]) == Counter(task["input"])
    )


def exposure(action):
    ops = [c["keystrokes"].strip() for c in action["commands"]]
    if ops == ["DESC", "POSITIVE", "SUBMIT"]:
        return 1
    if ops == ["DESC", "SUBMIT"]:
        return 0
    return None


def export(root, out):
    source, staging = root / "source", root / "staging"
    plan = read(root / "plan.json")
    m, rows = load_run(source)
    require(
        m["split"] == "source"
        and plan["tasks"] == m["tasks"]
        and [r["seed"] for r in rows] == plan["seeds"] == list(range(91000, 91006))
        and m["max_steps"] == 4
        and m["actor"]["max_new_tokens"] == 256
        and m["actor"]["temperature"] == 0
        and not m["actor"]["do_sample"],
        "wrong source plan or actor/turn budget",
    )
    item = next((r for r in rows if r["verifier"]["status"] == "fail"), None)
    require(item is not None, "no independently failed source episode")
    evidence = read(staging / "source-evidence.json")
    bank = read(staging / "bank.json")
    require(
        bank["bank_id"] == digest({k: v for k, v in bank.items() if k != "bank_id"})
        and bank["protocol"] == protocol_digest()
        and bank["evidence_index_digest"] == digest(evidence)
        and bank["source_manifest_digest"] == digest(m)
        and len(evidence) == 1,
        "staging lineage mismatch",
    )
    entry = evidence[0]
    proposal_name = entry["proposal"]
    require(Path(proposal_name).name == proposal_name, "invalid proposal path")
    p = read(source / "proposals" / proposal_name)
    require(
        p["run_id"] == m["run_id"]
        and p["episode_id"] == item["episode_id"] == entry["episode_id"]
        and p["model"] == p["response_model"] == "gpt-6-sol"
        and p["status"] == "ok"
        and digest(p) == entry["proposal_digest"]
        and digest(proposal_source(source, item)) == p["source_digest"]
        and digest(p["candidate"]) == CANDIDATE
        and entry.get("source_verified") is True,
        "teacher candidate or independent source evidence mismatch",
    )
    turn = p["candidate"]["step"]
    pre, draft, _ = restore(source, item, turn)
    require(
        entry["turn"] == turn
        and applicable(item["task"], pre["state"])
        and exposure(draft["a0"]) == 1
        and exposure(entry["replacement"]) == 0,
        "source/accredited action signature mismatch",
    )
    phases = entry["confirmations"]
    require([e["phase"] for e in phases] == ["discovery", "confirmation"], "missing source confirmation")
    ids = set()
    checks = []
    for phase, order in zip(phases, [["original", "repair"], ["repair", "original"]], strict=True):
        require(phase["order"] == order and phase["delta_success"] == 1, "invalid confirmation design")
        for side in order:
            result = phase["arms"][side]
            ref = f"replays/{digest(p)}/{phase['phase']}-{side}.json"
            trace = read(staging / ref)
            score = trace["score"]
            y = int(side == "repair")
            require(
                ref == result["evidence"]
                and digest(trace) == result["digest"]
                and score == result["verdict"]
                and score["reward"] == y
                and score["status"] == ("pass" if y else "fail")
                and trace["termination"] == result["termination"] == "submitted"
                and trace["final_state"]["submitted"] == result["submitted"]
                and score["request_id"] not in ids,
                "source replay or fresh receipt mismatch",
            )
            ids.add(score["request_id"])
            require(
                verify(item["task"], trace["final_state"])["reward"] == y,
                "independent terminal regrade failed",
            )
            executed = trace["turns"][turn - 1]["executed_action"]
            require(
                executed == (entry["replacement"] if y else draft["a0"])
                and len(trace["turns"]) <= m["max_steps"],
                "wrong repair/original or exceeded horizon",
            )
            checks.append(
                {
                    "phase": phase["phase"],
                    "side": side,
                    "score": score,
                    "path": "staging/" + ref,
                    "digest": digest(trace),
                    "tool_calls": trace["turns"][-1]["used_tool_calls"],
                }
            )
    state = pre["state"]
    fixtures = {
        "intact_complete_descending": applicable(item["task"], state),
        "positive_only_request": applicable(
            {**item["task"], "family": "positive", "instruction": "Keep only positive records."}, state
        ),
        "records_already_lost": applicable(item["task"], {**state, "working": state["working"][:-1]}),
        "research_original_exposure": exposure(draft["a0"]),
        "repair_exposure": exposure(entry["replacement"]),
    }
    require(
        list(fixtures.values()) == [True, False, False, 1, 0], "development predicate fixture failed"
    )
    card = {
        "schema": "mini-memory-development-v2",
        "status": "source_verified_and_predicate_checked",
        "contract": "full-descending-intact-records/v2",
        "candidate": p["candidate"],
        "source_run": m["run_id"],
        "source_episode": item["episode_id"],
        "checkpoint": draft["checkpoint_id"],
        "source_pre_digest": digest(pre),
        "original": draft["a0"],
        "repair": entry["replacement"],
        "source_plan_digest": digest(plan),
        "proposal_digest": digest(p),
        "proposal_path": "source/proposals/" + proposal_name,
        "teacher_usage": p["usage"],
        "teacher_seconds": p["seconds"],
        "source_manifest_digest": digest(m),
        "source_protocol_digest": bank["protocol"],
        "staging_bank_id": bank["bank_id"],
        "source_budget": {"turns": m["max_steps"], "max_new_tokens": m["actor"]["max_new_tokens"]},
        "observed_replays": checks,
        "source_predicate_checks": fixtures,
        "scope": "Single verified source state and composite action. Prospective target intervention value untested. "
        "Independent development package; not a replacement or retrospective extension of the pilot bank.",
    }
    out.mkdir(parents=True, exist_ok=False)
    save(out / "card.json", {**card, "memory_id": digest(card)})
    print(out / "card.json")


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--root", required=True, type=Path)
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    export(args.root, args.out)


if __name__ == "__main__":
    main()
