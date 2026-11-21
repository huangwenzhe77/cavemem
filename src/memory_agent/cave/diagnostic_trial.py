"""Fresh mini cohorts for live policy execution and a separate randomized two-stage inspection pilot."""

import argparse
import hashlib
import hmac
import random
import secrets
from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import monotonic

from . import formal, formal_plan as fp, gate, gate_validation as gv, single_arm as online
from .constrained_gate import Policy
from .models import FrozenQwen, xml_parser
from .records import digest, read
from .replay import require
from .supplement_plan import SHAPES, make_input


def draw(key, eid, stage):
    # Independent streams. Use 53 bits so conversion to float can never yield 1.
    b = hmac.digest(bytes.fromhex(key), (stage + "/" + eid).encode(), hashlib.sha256)
    return (int.from_bytes(b[:8], "big") >> 11) / 2**53


def diagnostic(state, action):
    """Read-only static check for cardinality loss in the proposed mini DSL batch."""
    working = tuple(state["working"])
    before = len(working)
    for c in action["commands"]:
        op = c["keystrokes"].strip()
        if op == "POSITIVE":
            working = tuple(v for v in working if v > 0)
        elif op == "UNIQUE":
            working = tuple(dict.fromkeys(working))
    return {"would_remove_records": len(working) < before}


def register(root, outputs, model):
    parent = outputs / "cave-single-arm-01"
    plan, pid = fp.load(parent)
    policy = gate.signed(model, "policy_id")
    require(policy["bank_id"] == plan["bank_id"], "bank drift")
    seen = set()
    for path in sorted(outputs.glob("cave-*/**/preregistration.json")):
        old = read(path)
        for cohort in old.get("cohorts", []):
            seen.update(tuple(sorted(r["task"]["input"])) for r in cohort.get("tasks", [])
                        if isinstance(r, dict) and "input" in r.get("task", {}))
    for path in sorted(outputs.glob("cave-*/**/manifest.json")):
        old = read(path)
        if old.get("benchmark") == "mini-data-v1":
            seen.update(tuple(sorted(t["input"])) for t in old["tasks"])
    sealed = read(parent/"sealed-test.json")
    require(digest(sealed) == plan["test_manifest_digest"], "test seal changed")
    seen.update(tuple(sorted(r["task"]["input"])) for r in sealed["tasks"])
    tasks = []
    seed = 4100000
    for phase, per_template in (("inspect_explore", 12), ("binary_live", 3)):
        for name, (family, text) in gv.TEMPLATES.items():
            for k in range(per_template):
                while True:
                    seed += 1
                    values = make_input(seed, SHAPES[k % len(SHAPES)], family)
                    if values is not None and not any(fp.near_duplicate(values, t) for t in seen):
                        break
                seen.add(tuple(sorted(values)))
                task = {"template_id": name, "family": family, "instruction": text, "input": values}
                tasks.append({"episode_id": f"ep-{len(tasks):04d}", "phase": phase, "task": task,
                              "task_digest": digest(task), "seed": seed, "main_opportunity": 1, "q": 1.})
    root.mkdir(exist_ok=False)
    reg = {"parent_plan": pid, "policy_id": policy["policy_id"], "schema": "mini-first-opportunity-inspect/v1",
           "code": fp.code_digest("diagnostic_trial.py"), "online_code": fp.code_digest("single_arm.py"),
           "tasks": tasks, "seed_hex": secrets.token_hex(32),
           "stage1": ["silent", "remind", "inspect"], "stage2": ["silent", "remind"],
           "probe_budget": 1, "cohorts": [{"split": s, "tasks": [r for r in tasks if r["phase"] == s]}
                                         for s in ("inspect_explore", "binary_live")],
           "design": "First executable proposal only. Common eligibility and budget. Explore independent "
           "uniform stage 1 and 2; binary_live samples frozen KL policy. No replacement or retries."}
    online.durable(root/"preregistration.json", {**reg, "plan_id": digest(reg)})


class TrialEpisode(online.Episode):
    def __init__(self, *args, reg, policy):
        super().__init__(*args)
        self.reg, self.policy = reg, policy

    def select(self, folder, pre, draft):
        self.selected_turn, self.checkpoint = pre["step"], pre["checkpoint_id"]
        start = monotonic()
        match = gv.route(self.cards, self.row["task"], pre, draft)
        self.usage["matching_seconds"] += monotonic()-start
        self.main_status = match["status"]
        if match.get("rho") != 1:
            return None, None
        start = monotonic()
        x = formal.gate_inputs(self.row, pre, draft, match, self.plan)
        msg = deepcopy(self.messages)
        msg[-1]["content"] += "\n\nAdvisory note for this proposal only: " + match["reminder"]
        self.usage["matching_seconds"] += monotonic()-start
        n = self.prompt_tokens(msg)
        exploring = self.row["phase"] == "inspect_explore"
        # Common opportunity and one diagnostic tool slot reserved for every exploration arm.
        if not self.fits(n) or self.usage["tool_calls"] + len(draft["a0"]["commands"]) + int(exploring) > self.plan["total_tools"]:
            self.main_status = "insufficient_common_budget"
            return None, None
        sel = {"phase": self.row["phase"], "checkpoint": self.checkpoint, "features": x,
               "draft_digest": digest(draft), "pre_digest": digest(pre), "plan_id": self.pid}
        online.durable(folder/"selection.json", sel)
        start = monotonic()
        key, eid = self.reg["seed_hex"], self.row["episode_id"]
        u = draw(key, eid, "stage1")
        if exploring:
            choices = self.reg["stage1"]
            first = choices[int(u*len(choices))]
            probs = {a: 1/len(choices) for a in choices}
        else:
            p = self.policy.probability(x)
            first = "remind" if u < p else "silent"
            probs = {"remind": p, "silent": 1-p}
        self.usage["assignment_seconds"] += monotonic()-start
        receipt = {"first": first, "b1": probs, "u1": u, "selection_digest": digest(sel),
                   "policy_id": self.policy.doc["policy_id"], "checkpoint": self.checkpoint}
        online.durable(folder/"first-stage.json", receipt)
        arm = first
        if first == "inspect":
            start = monotonic()
            observation = diagnostic(self.state, draft["a0"])
            seconds = monotonic()-start
            self.usage["tool_seconds"] += seconds
            self.usage["tool_calls"] += 1
            probe = {"observation": observation, "tool_calls": 1, "tool_seconds": seconds,
                     "first_stage_digest": digest(receipt)}
            online.durable(folder/"probe.json", probe)
            start = monotonic()
            u2 = draw(key, eid, "stage2")
            arm = "remind" if u2 < .5 else "silent"
            self.usage["assignment_seconds"] += monotonic()-start
            online.durable(folder/"second-stage.json", {"arm": arm, "b2": .5, "u2": u2,
                           "probe_digest": digest(probe)})
        self.assigned = {"arm": arm, "first": first, "checkpoint_id": self.checkpoint}
        self.main_status = "assigned"
        return msg, n


def run(root, outputs, model):
    reg = gate.signed(root/"preregistration.json", "plan_id")
    plan, pid, _, cards = online.validate(outputs/"cave-single-arm-01", fp.DEFAULT_BANK, fp.DEFAULT_PACKAGE)
    policy = Policy(model, outputs/"cave-mlp-01/frozen/model.json")
    require(pid == reg["parent_plan"] and reg["code"] == fp.code_digest("diagnostic_trial.py")
            and reg["online_code"] == fp.code_digest("single_arm.py") and reg["policy_id"] == policy.doc["policy_id"],
            "registration drift")
    with online.guard(root/"run.lock"):
        actor = parser = None
        index = []
        raw = root/"episodes"
        for r in reg["tasks"]:
            path = raw/r["episode_id"]
            if not (path/"summary.json").exists():
                if path.exists():
                    # Preserve all observed assignments after interruption. Never repeat a started episode.
                    s = {"termination": "interrupted", "main_status": "interrupted", "episode_id": r["episode_id"],
                         "usage": {"observed_utility": None}, "verifier": {"reward": None}}
                else:
                    if actor is None:
                        actor = FrozenQwen(plan["actor"]["actor"], plan["actor"]["max_new_tokens"], 0)
                        require(actor.info == plan["actor"], "frozen actor changed")
                        parser = xml_parser()
                    s = TrialEpisode(path, plan, reg["plan_id"], reg["plan_id"], r, actor, parser, cards, None,
                                     reg=reg, policy=policy).run()
                online.durable(path/"summary.json", s)
            s = read(path/"summary.json")
            index.append({"id": r["episode_id"], "summary": digest(s), "events":
                          {p.relative_to(path).as_posix(): digest(read(p)) for p in sorted(path.rglob("*.json"))}})
            print(r["episode_id"], r["phase"], s["main_status"], flush=True)
        seal = {"plan_id": reg["plan_id"], "rows": index, "events_digest": digest(index)}
        if not (root/"seal.json").exists():
            online.durable(root/"seal.json", seal)
        require(read(root/"seal.json") == seal, "trial records changed")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("mode", choices=("register", "collect"))
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--outputs", type=Path, default=fp.OUTPUT)
    cli.add_argument("--model", type=Path, default=fp.OUTPUT/"cave-primal-dual-01/model.json")
    args = cli.parse_args()
    (register if args.mode == "register" else run)(args.root, args.outputs, args.model)
