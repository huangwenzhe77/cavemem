"""Prospective single-trajectory randomization before the scheduled action executes."""

import argparse
import hashlib
import hmac
import math
import os
import random
import secrets
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from . import formal, formal_plan as fp, gate_validation, mini
from .experiment import open_bank
from .models import FrozenQwen, SYSTEM, xml_parser
from .records import digest, encode, read, verify
from .replay import advance, protocol_digest, require


PROBS = {"silent": 0.5, "remind": 0.5}
STREAM = b"CAVE-online-single-arm/v1\0"


def durable(path, obj):
    """Flush before returning; an assignment must survive before the arm may run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "event already recorded: " + path.name)
    tmp = path.parent / ("." + path.name + "." + uuid.uuid4().hex)
    with tmp.open("xb") as f:
        f.write((encode(obj) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    if os.name == "posix":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def assign(secret, pid, eid):
    mac = hmac.new(
        secret,
        STREAM + pid.encode("ascii") + b"\0" + eid.encode("ascii"),
        hashlib.sha256,
    ).digest()
    n = int.from_bytes(mac[:8], "big")
    arm = "remind" if n < 2**63 else "silent"
    return {
        "arm": arm,
        "Z": int(arm == "remind"),
        "propensities": PROBS,
        "observed_propensity": 0.5,
        "uniform_numerator": str(n),
        "uniform_denominator": str(2**64),
        "hmac_sha256": mac.hex(),
    }


def validate(root, bank_path, package):
    plan = read(root / "preregistration.json")
    pid = plan.pop("plan_id")
    require(
        pid == digest(plan)
        and plan["schema"] == "randomized-single-arm/v1"
        and plan["protocol"] == protocol_digest()
        and digest(read(root / "sealed-test.json")) == plan["test_manifest_digest"]
        and plan["assignment_probabilities"] == PROBS
        and plan["online_templates"]
        == {k: list(v) for k, v in gate_validation.TEMPLATES.items()}
        and [c["split"] for c in plan["cohorts"]] == ["single_arm"]
        and plan["actor"]["max_new_tokens"] == 256
        and plan["actor"]["temperature"] == 0
        and not plan["actor"]["do_sample"]
        and plan["max_steps"] == 4
        and plan["total_tokens"] > 0
        and plan["total_tools"] > 0,
        "wrong online study",
    )
    rows = plan["cohorts"][0]["tasks"]
    require(
        len(rows) > 0
        and len({r["task_digest"] for r in rows}) == len(rows)
        and len({r["seed"] for r in rows}) == len(rows),
        "duplicate tasks/seeds",
    )
    for i, r in enumerate(rows):
        t = r["task"]
        family, instruction = gate_validation.TEMPLATES[t["template_id"]]
        require(
            r["episode_id"] == f"ep-{i:04d}"
            and (t["family"], t["instruction"]) == (family, instruction)
            and digest(t) == r["task_digest"]
            and r["q"] == 1 / plan["max_steps"]
            and r["main_opportunity"]
            == random.Random("checkpoint:" + str(r["seed"])).randint(
                1, plan["max_steps"]
            ),
            "task or ex ante opportunity changed",
        )
    parent, old, _ = open_bank(bank_path)
    desc = read(package / "development-v2/card.json")
    require(
        len(old) == 1
        and parent == plan["parent_bank_id"]
        and desc["memory_id"] == plan["descending_memory_id"]
        and desc["memory_id"]
        == digest({k: v for k, v in desc.items() if k != "memory_id"})
        and digest(
            {
                "schema": "mini-prospective-bank-v2",
                "parent_bank_id": parent,
                "descending_memory_id": desc["memory_id"],
            }
        )
        == plan["bank_id"],
        "frozen source card digest mismatch",
    )
    return plan, pid, rows, {"ascending": old[0], "descending": desc}


def parse_response(parser, text):
    parsed = parser.parse_response(text)
    commands = [asdict(c) for c in parsed.commands]
    error = parsed.error
    if any(not math.isfinite(c["duration"]) for c in commands):
        commands, error = [], "nonfinite duration"
    return (
        {"commands": commands, "complete": parsed.is_task_complete},
        error,
        parsed.warning,
    )


class Episode:
    def __init__(self, path, plan, pid, rid, row, actor, parser, cards, secret):
        self.path, self.plan, self.pid, self.rid, self.row = path, plan, pid, rid, row
        self.actor, self.parser, self.cards, self.secret = actor, parser, cards, secret
        self.state = mini.initial(row["task"])
        self.messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": row["task"]["instruction"]
                + "\nCurrent working list: "
                + str(self.state["working"]),
            },
        ]
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "actor_calls": 0,
            "revision_calls": 0,
            "tool_calls": 0,
            "actor_seconds": 0.0,
            "actor_request_wall_seconds": 0.0,
            "tool_seconds": 0.0,
            "matching_seconds": 0.0,
            "assignment_seconds": 0.0,
            "prompt_count_seconds": 0.0,
            "usage_complete": True,
        }
        self.opportunities = 0
        self.main_status = "never_reached"
        self.checkpoint = None
        self.assigned = None
        self.selected_turn = None
        self.steps = []
        self.suffix_seed = random.Random(
            f"coupling:{plan['seed']}:single_arm:{row['episode_id']}:0"
        ).randrange(2**31)

    def prompt_tokens(self, msg):
        start = time.monotonic()
        try:
            return self.actor.count_prompt_tokens(msg)
        finally:
            self.usage["prompt_count_seconds"] += time.monotonic() - start

    def fits(self, n):
        u = self.usage
        return (
            u["actor_calls"] < self.plan["max_steps"]
            and u["input_tokens"]
            + u["output_tokens"]
            + n
            + self.plan["actor"]["max_new_tokens"]
            <= self.plan["total_tokens"]
        )

    def generate(self, folder, stage, messages, seed, n):
        u = self.usage
        durable(
            folder / (stage + "-request.json"),
            {
                "stage": stage,
                "messages": messages,
                "decode_seed": seed,
                "prompt_tokens": n,
                "usage_before": dict(u),
            },
        )
        u["actor_calls"] += 1
        u["revision_calls"] += int(stage == "revision")
        start = time.monotonic()
        try:
            gen = self.actor.generate(messages, seed)
            require(
                gen["prompt_tokens"] == n
                and 0
                <= gen["completion_tokens"]
                <= self.plan["actor"]["max_new_tokens"]
                and math.isfinite(gen["seconds"])
                and gen["seconds"] >= 0,
                "actor accounting drift",
            )
        except Exception as exc:
            u["usage_complete"] = False
            durable(
                folder / (stage + "-error.json"), {"error_type": type(exc).__name__}
            )
            return None
        finally:
            u["actor_request_wall_seconds"] += time.monotonic() - start
        u["input_tokens"] += gen["prompt_tokens"]
        u["output_tokens"] += gen["completion_tokens"]
        u["actor_seconds"] += gen["seconds"]
        durable(folder / (stage + "-generation.json"), gen)
        return gen

    def select(self, folder, pre, draft):
        self.checkpoint = pre["checkpoint_id"]
        self.selected_turn = pre["step"]
        start = time.monotonic()
        d = gate_validation.route(self.cards, self.row["task"], pre, draft)
        x = (
            formal.gate_inputs(self.row, pre, draft, d, self.plan)
            if d.get("memory_id")
            else None
        )
        self.usage["matching_seconds"] += time.monotonic() - start
        msg = deepcopy(self.messages)
        n = None
        if d.get("rho") == 1:
            msg[-1]["content"] += (
                "\n\nAdvisory note for this proposal only: " + d["reminder"]
            )
            n = self.prompt_tokens(msg)
            affordable = (
                self.fits(n)
                and self.usage["tool_calls"] + len(draft["a0"]["commands"])
                <= self.plan["total_tools"]
            )
            self.main_status = (
                "eligible" if affordable else "insufficient_revision_budget"
            )
        else:
            self.main_status = d["status"]
        sel = {
            "plan_id": self.pid,
            "run_id": self.rid,
            "episode_id": self.row["episode_id"],
            "checkpoint_id": self.checkpoint,
            "turn": self.selected_turn,
            "opportunity": self.opportunities,
            "q": self.row["q"],
            "pre_digest": digest(pre),
            "draft_digest": digest(draft),
            "a0_digest": draft["a0_digest"],
            "features": x,
            "decision": d,
            "eligibility": self.main_status,
            "usage_after_draft": dict(self.usage),
            "revision_prompt_tokens": n,
        }
        durable(folder / "selection.json", sel)
        if self.main_status == "eligible":
            start = time.monotonic()
            result = assign(self.secret, self.pid, self.row["episode_id"])
            self.usage["assignment_seconds"] += time.monotonic() - start
            receipt = {
                **result,
                "episode_id": self.row["episode_id"],
                "plan_id": self.pid,
                "run_id": self.rid,
                "checkpoint_id": self.checkpoint,
                "selection_digest": digest(sel),
                "drawn_before_execution": True,
            }
            durable(folder / "assignment.json", receipt)
            self.assigned = receipt
            self.main_status = "assigned"
        return msg, n

    def step(self, turn):
        r, u = self.row, self.usage
        cp = r["episode_id"] + f"/{turn:03d}"
        folder = self.path / f"{turn:03d}"
        suffix = self.assigned is not None
        stage = "suffix" if suffix else "draft"
        seed = random.Random(
            f"{self.suffix_seed}:suffix:{turn - self.selected_turn}"
            if suffix
            else f"actor:{r['seed']}:{turn}"
        ).randrange(2**31)
        pre = {
            "checkpoint_id": cp,
            "step": turn,
            "messages": self.messages,
            "state": self.state,
            "decode_seed": seed,
            "remaining_steps": self.plan["max_steps"] - turn + 1,
            "used_input_tokens": u["input_tokens"],
            "used_output_tokens": u["output_tokens"],
            "used_tool_calls": u["tool_calls"],
        }
        durable(folder / "pre.json", pre)
        n = self.prompt_tokens(self.messages)
        if not self.fits(n):
            return "actor_budget"
        gen = self.generate(folder, stage, self.messages, seed, n)
        if gen is None:
            return "actor_error"
        action, error, warning = parse_response(self.parser, gen["text"])
        executable = bool(action["commands"]) and not error
        self.opportunities += int(executable)
        draft = {
            "checkpoint_id": cp,
            "messages": self.messages + [{"role": "assistant", "content": gen["text"]}],
            "a0": action,
            "a0_digest": digest(action),
            "error": error,
            "warning": warning,
            "used_input_tokens": u["input_tokens"],
            "used_output_tokens": u["output_tokens"],
            "used_tool_calls": u["tool_calls"],
        }
        durable(folder / "draft.json", draft)
        self.steps.append(
            {
                "turn": turn,
                "checkpoint_id": cp,
                "executable": executable,
                "opportunity": self.opportunities if executable else None,
            }
        )
        if (
            self.checkpoint is None
            and executable
            and self.opportunities == r["main_opportunity"]
        ):
            msg, n = self.select(folder, pre, draft)
            if self.assigned and self.assigned["arm"] == "remind":
                seed = random.Random(f"{self.suffix_seed}:revision").randrange(2**31)
                gen = self.generate(folder, "revision", msg, seed, n)
                if gen is None:
                    return "actor_error"
                action, error, _ = parse_response(self.parser, gen["text"])
                stage = "revision"
        if u["tool_calls"] + len(action["commands"]) > self.plan["total_tools"]:
            return "tool_budget"
        durable(
            folder / "chosen-action.json",
            {"stage": stage, "action": action, "parse_error": error},
        )
        start = time.monotonic()
        obs, executed, end = advance(self.state, action, error)
        elapsed = time.monotonic() - start
        u["tool_seconds"] += elapsed
        u["tool_calls"] += len(executed["commands"]) if executed else 0
        self.messages = self.messages + [
            {"role": "assistant", "content": gen["text"]},
            {"role": "user", "content": "\n".join(obs)},
        ]
        durable(
            folder / "execution.json",
            {
                "stage": stage,
                "executed_action": executed,
                "observation": obs,
                "state": self.state,
                "tool_seconds": elapsed,
                "usage": dict(u),
            },
        )
        return end

    def run(self):
        started = time.monotonic()
        durable(
            self.path / "started.json",
            {
                "task_digest": self.row["task_digest"],
                "episode_id": self.row["episode_id"],
                "main_opportunity": self.row["main_opportunity"],
                "q": self.row["q"],
                "run_id": self.rid,
                "plan_id": self.pid,
            },
        )
        termination = "budget"
        try:
            for turn in range(1, self.plan["max_steps"] + 1):
                end = self.step(turn)
                if end:
                    termination = end
                    break
        except (OSError, KeyboardInterrupt):
            raise
        except Exception as exc:
            termination = "infrastructure_error"
            self.usage["usage_complete"] = False
            durable(self.path / "error.json", {"error_type": type(exc).__name__})
        score = (
            {"status": "unavailable", "reward": None, "reason": termination}
            if termination in ("actor_error", "infrastructure_error")
            else verify(self.row["task"], self.state)
        )
        durable(self.path / "verdict.json", score)
        u = self.usage
        u["total_tokens"] = (
            u["input_tokens"] + u["output_tokens"] if u["usage_complete"] else None
        )
        u["accounted_seconds"] = (
            sum(
                u[k]
                for k in (
                    "actor_seconds",
                    "tool_seconds",
                    "matching_seconds",
                    "assignment_seconds",
                    "prompt_count_seconds",
                )
            )
            if u["usage_complete"]
            else None
        )
        u["episode_wall_seconds"] = time.monotonic() - started
        cost = (
            (
                self.plan["lambda_tokens"] * u["total_tokens"] / 1000
                + self.plan["lambda_tool"] * u["tool_calls"]
                + self.plan["lambda_time"] * u["accounted_seconds"] / 60
            )
            if u["usage_complete"]
            else None
        )
        u["cost_penalty"] = cost
        u["observed_utility"] = (
            score["reward"] - cost
            if score["reward"] is not None and cost is not None
            else None
        )
        return self.summary(termination, score)

    def summary(self, end, score):
        return {
            "plan_id": self.pid,
            "run_id": self.rid,
            "episode_id": self.row["episode_id"],
            "task_digest": self.row["task_digest"],
            "seed": self.row["seed"],
            "q": self.row["q"],
            "main_opportunity": self.row["main_opportunity"],
            "main_status": self.main_status,
            "opportunities": self.opportunities,
            "checkpoint_id": self.checkpoint,
            "steps": self.steps,
            "assigned_arm": self.assigned["arm"] if self.assigned else None,
            "assignment": self.assigned,
            "usage": self.usage,
            "termination": end,
            "verifier": score,
        }


def interrupted(path, row, pid, rid):
    # A started task is never rerun. Preserve any durable assignment under ITT.
    sels = list(path.glob("*/selection.json"))
    draws = list(path.glob("*/assignment.json"))
    require(len(sels) <= 1 and len(draws) <= len(sels), "invalid partial chronology")
    sel = read(sels[0]) if sels else None
    draw = read(draws[0]) if draws else None
    score = (
        read(path / "verdict.json")
        if (path / "verdict.json").exists()
        else {"status": "unavailable", "reward": None, "reason": "interrupted"}
    )
    return {
        "plan_id": pid,
        "run_id": rid,
        "episode_id": row["episode_id"],
        "seed": row["seed"],
        "task_digest": row["task_digest"],
        "q": row["q"],
        "main_opportunity": row["main_opportunity"],
        "main_status": "assigned"
        if draw
        else (sel["eligibility"] if sel else "interrupted_before_selection"),
        "checkpoint_id": sel["checkpoint_id"] if sel else None,
        "assignment": draw,
        "assigned_arm": draw["arm"] if draw else None,
        "steps": None,
        "opportunities": None,
        "termination": "interrupted",
        "verifier": score,
        "usage": {
            "usage_complete": False,
            "total_tokens": None,
            "accounted_seconds": None,
            "cost_penalty": None,
            "observed_utility": None,
        },
    }


@contextmanager
def guard(path):
    with path.open("a+b") as f:
        if os.name == "posix":
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        yield


def collect(args):
    plan, pid, rows, cards = validate(args.root, args.parent_bank, args.source_package)
    run = args.root / "single_arm"
    key_path = args.root.parent / ("." + args.root.name + ".assignment-key")
    if not args.resume:
        run.mkdir()
    require(run.is_dir(), "no existing run to resume")
    with guard(run / "runner.lock"):
        if not (run / "manifest.json").exists():
            require(
                not any(run.glob("ep-*")), "cannot initialize over trajectory records"
            )
            if not key_path.exists():
                with os.fdopen(
                    os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
                ) as f:
                    f.write(secrets.token_bytes(32))
                    f.flush()
                    os.fsync(f.fileno())
            elif not args.resume:
                raise ValueError("a private key already exists for this output root")
            secret = key_path.read_bytes()
            durable(
                run / "manifest.json",
                {
                    "schema": "online-single-arm-run/v1",
                    "split": "single_arm",
                    "plan_id": pid,
                    "run_id": uuid.uuid4().hex,
                    "commitment_sha256": hashlib.sha256(secret).hexdigest(),
                    "assignment_probabilities": PROBS,
                    "collector_code": fp.code_digest("single_arm.py"),
                    "route_code": fp.code_digest("gate_validation.py"),
                    "actor": plan["actor"],
                    "bank_id": plan["bank_id"],
                    "episode_ids": [r["episode_id"] for r in rows],
                },
            )
        mf = read(run / "manifest.json")
        secret = key_path.read_bytes()
        require(
            len(secret) == 32
            and hashlib.sha256(secret).hexdigest() == mf["commitment_sha256"]
            and mf["plan_id"] == pid
            and mf["collector_code"] == fp.code_digest("single_arm.py")
            and mf["route_code"] == fp.code_digest("gate_validation.py"),
            "private seed, run or code changed",
        )
        actor = parser = None
        records = []
        for row in rows:
            folder = run / row["episode_id"]
            file = folder / "summary.json"
            if file.is_file():
                records.append(read(file))
                continue
            if folder.exists():
                res = interrupted(folder, row, pid, mf["run_id"])
            else:
                if actor is None:
                    actor = FrozenQwen(
                        plan["actor"]["actor"], plan["actor"]["max_new_tokens"], 0
                    )
                    require(
                        actor.info == mf["actor"],
                        "live actor differs from frozen configuration",
                    )
                    parser = xml_parser()
                res = Episode(
                    folder, plan, pid, mf["run_id"], row, actor, parser, cards, secret
                ).run()
            res["event_digests"] = {
                p.relative_to(folder).as_posix(): digest(read(p))
                for p in sorted(folder.rglob("*.json"))
            }
            durable(file, res)
            records.append(res)
            print(
                row["episode_id"],
                res["main_status"],
                res["assigned_arm"],
                res["termination"],
                flush=True,
            )
        for row, res in zip(rows, records, strict=True):
            folder = run / row["episode_id"]
            require(
                res["episode_id"] == row["episode_id"]
                and res["task_digest"] == row["task_digest"]
                and res["plan_id"] == pid
                and res["run_id"] == mf["run_id"]
                and res["event_digests"]
                == {
                    p.relative_to(folder).as_posix(): digest(read(p))
                    for p in sorted(folder.rglob("*.json"))
                    if p != folder / "summary.json"
                },
                "changed online episode",
            )
        index = run / "index.jsonl"
        body = "".join(encode(r) + "\n" for r in records)
        if not index.exists():
            tmp = run / "index.pending"
            with tmp.open("wb") as f:
                f.write(body.encode())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, index)
        require(index.read_text(encoding="utf-8") == body, "online index changed")
        seal = {
            "plan_id": pid,
            "manifest_digest": digest(mf),
            "index_digest": digest(records),
            "episodes": len(records),
        }
        if not (run / "seal.json").exists():
            durable(run / "seal.json", seal)
        require(read(run / "seal.json") == seal, "online seal changed")
        reveal = {
            "plan_id": pid,
            "seal_digest": digest(seal),
            "secret_hex": secret.hex(),
            "commitment_sha256": mf["commitment_sha256"],
        }
        if not (run / "randomization-reveal.json").exists():
            durable(run / "randomization-reveal.json", reveal)
        require(read(run / "randomization-reveal.json") == reveal, "invalid reveal")
    print(run / "seal.json")


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("command", choices=["collect"])
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument(
        "--resume",
        action="store_true",
        help="Keep any interrupted started task missing; continue only untouched tasks.",
    )
    cli.add_argument("--parent-bank", type=Path, default=fp.DEFAULT_BANK)
    cli.add_argument("--source-package", type=Path, default=fp.DEFAULT_PACKAGE)
    collect(cli.parse_args())


if __name__ == "__main__":
    main()
