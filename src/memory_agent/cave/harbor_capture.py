"""Opt-in instrumentation of the vendored Terminus2 loop; ordinary PMA uses its original class."""
import random
import hashlib
import json
import shlex
import time
import uuid
from pathlib import Path

from harbor.agents.terminus_2.terminus_2 import Terminus2
from .records import digest, save


class CapturedTerminus2(Terminus2):
    def __init__(self, *, logs_dir, max_turns, capture_seed, **kwargs):
        super().__init__(logs_dir=logs_dir, max_turns=max_turns, **kwargs)
        self.capture_root = Path(logs_dir) / "cave"
        self.capture_root.mkdir(parents=True, exist_ok=False)
        self.steps = []
        main_step = random.Random(capture_seed).randint(1, max_turns)
        self.opportunities = 0
        self.schedule = {"main_opportunity": main_step, "q": 1/max_turns, "seed": capture_seed,
                         "max_turns": max_turns, "benchmark": "harbor",
                         "memory_enabled": False, "teacher_enabled": False,
                         "restore_status": "unvalidated", "actor": self._model_name,
                         "parser": type(self._parser).__name__}
        self._index()

    def _index(self):
        main = next((s for s in self.steps if s["opportunity_index"] == self.schedule["main_opportunity"]), None)
        save(self.capture_root / "index.json", {**self.schedule, "steps": self.steps,
             "main_status": main["status"] if main else "never_reached"})

    async def _handle_llm_interaction(self, chat, prompt, logging_paths, original_instruction="", session=None):
        step = len(self.steps) + 1
        self.current_checkpoint = self.capture_root / f"{step:03d}"
        pre = {"step": step, "messages": chat.messages, "prompt": prompt,
               "pending_completion": self._pending_completion, "remaining_steps": self._max_episodes-step+1,
               "input_tokens": chat.total_input_tokens, "output_tokens": chat.total_output_tokens,
               "environment_restore": "unvalidated", "memory_enabled": False, "teacher_enabled": False}
        save(self.current_checkpoint / "pre.json", pre)
        self.steps.append({"step": step, "turn_id": step, "opportunity_index": None,
                           "checkpoint_id": self.current_checkpoint.name,
                           "status": "actor_pending"})
        self._index()
        start = time.monotonic()
        try:
            result = await super()._handle_llm_interaction(chat, prompt, logging_paths, original_instruction, session)
        except BaseException as exc:
            self.steps[-1].update(status="actor_error", error_type=type(exc).__name__)
            self._index()
            raise
        commands, complete, feedback, _, _, response = result
        self.cached = {"commands": [self._command(c) for c in commands], "complete": complete}
        save(self.current_checkpoint / "draft.json", {"a0": self.cached, "a0_digest": digest(self.cached),
             "messages": chat.messages, "text": response.content, "feedback": feedback,
             "remaining_steps": self._max_episodes-step, "input_tokens": chat.total_input_tokens,
             "output_tokens": chat.total_output_tokens, "seconds": time.monotonic()-start})
        self.steps[-1]["status"] = "parse_error" if "ERROR:" in feedback else "draft_cached"
        if commands and "ERROR:" not in feedback:
            self.opportunities += 1
            self.steps[-1]["opportunity_index"] = self.opportunities
        self._index()
        return result

    @staticmethod
    def _command(cmd):
        return {"keystrokes": cmd.keystrokes, "duration": cmd.duration_sec}

    async def _execute_commands(self, commands, session):
        batch = [self._command(c) for c in commands]
        if batch != self.cached["commands"]:
            raise RuntimeError("execution differs from the cached unassisted draft")
        try:
            result = await super()._execute_commands(commands, session)
        except BaseException as exc:
            self.steps[-1].update(status="execution_error", error_type=type(exc).__name__)
            self._index()
            raise
        save(self.current_checkpoint / "execution.json", {"dispatched_action": self.cached,
             "executed_action": None if result[0] else self.cached,
             "timeout": result[0], "observation": result[1]})
        self.steps[-1]["status"] = "partial_timeout" if result[0] else "executed"
        self._index()
        return result


async def verify_harbor(environment, task):
    """Run this task's verifier after actor termination in its isolated container."""
    request_id = uuid.uuid4().hex
    try:
        if not task.tests_dir or not (Path(task.tests_dir) / "test.sh").is_file():
            raise ValueError("missing trusted verifier")
        tests = Path(task.tests_dir)
        # A pre-existing reward can never satisfy this invocation.
        # The caller checked that the image did not expose /tests to the actor.
        # Refuse leftovers rather than overlaying code from another invocation.
        clean = await environment.exec("test ! -e /tests && test ! -L /tests && mkdir /tests && "
                                       "mkdir -p /logs/verifier && rm -f /logs/verifier/reward.txt "
                                       "/logs/verifier/ctrf.json",
                                       timeout_sec=20)
        if clean.return_code:
            raise RuntimeError("could not clear old reward")
        await environment.upload_dir(tests, "/tests")
        # Enroot.upload_dir does not propagate the internal extraction status.
        for path in tests.rglob("*"):
            if path.is_symlink():
                raise ValueError("unsupported symlink in trusted test bundle")
            if path.is_file():
                name = shlex.quote("/tests/" + path.relative_to(tests).as_posix())
                expected = hashlib.sha256(path.read_bytes()).hexdigest()
                check = await environment.exec(f"sha256sum -- {name}", timeout_sec=20)
                if check.return_code or not check.stdout or check.stdout.split()[0] != expected:
                    raise ValueError("trusted verifier upload mismatch")
        ran = await environment.exec("bash /tests/test.sh", timeout_sec=task.max_test_timeout_sec)
        got = await environment.exec("cat /logs/verifier/reward.txt", timeout_sec=20)
        if ran.return_code != 0 or got.return_code != 0:
            raise RuntimeError("verifier did not complete normally")
        reward = float(got.stdout.strip())
        if reward not in (0, 1):
            raise ValueError("expected binary Terminal-Bench reward")
        evidence = {}
        if "/logs/verifier/ctrf.json" in (tests / "test.sh").read_text(encoding="utf-8"):
            report = await environment.exec("cat /logs/verifier/ctrf.json", timeout_sec=20)
            if report.return_code or not report.stdout:
                raise ValueError("verifier generated no fresh test report")
            results = json.loads(report.stdout)["results"]
            cases = results["tests"]
            statuses = [c["status"] for c in cases]
            summary = results["summary"]
            if (not cases or summary["tests"] != len(cases)
                    or any(status not in {"passed", "failed"} for status in statuses)
                    or summary["passed"] != statuses.count("passed")
                    or summary["failed"] != statuses.count("failed")
                    or bool(reward) != all(s == "passed" for s in statuses)):
                raise ValueError("verifier test evidence does not substantiate the reward")
            evidence = {"test_count": len(cases), "report_digest": digest(results)}
        return {"request_id": request_id, "status": "pass" if reward else "fail", "reward": reward, **evidence}
    except Exception as exc:
        return {"request_id": request_id, "status": "unavailable", "reward": None,
                "error_type": type(exc).__name__, "reason": str(exc)[:300]}
