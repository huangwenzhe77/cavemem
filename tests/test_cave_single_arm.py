import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memory_agent.cave import single_arm as s


def xml(*ops):
    return (
        "<response><analysis></analysis><plan></plan><commands>"
        + "".join(f'<keystrokes duration="0">{op}</keystrokes>' for op in ops)
        + "</commands></response>"
    )


PLAN = {
    "max_steps": 4,
    "actor": {"actor": "fixture", "max_new_tokens": 256},
    "total_tools": 64,
    "total_tokens": 16384,
    "seed": 82,
    "lambda_tokens": 0.01,
    "lambda_tool": 0.001,
    "lambda_time": 0.01,
    "bank_id": "fixture",
}
CARDS = {
    "descending": {
        "memory_id": "fixture",
        "contract": "full-descending-intact-records/v2",
        "candidate": {"reminder": "Keep every number."},
    }
}
TASK = {
    "template_id": "batch-down",
    "family": "descending",
    "instruction": s.gate_validation.TEMPLATES["batch-down"][1],
    "input": [-2, 1, 1],
}


class Script:
    info = PLAN["actor"]

    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def count_prompt_tokens(self, messages):
        return 20

    def generate(self, messages, seed):
        self.calls.append(messages)
        answer = next(self.answers)
        if isinstance(answer, BaseException):
            raise answer
        return {
            "text": answer,
            "seconds": 0.01,
            "prompt_tokens": 20,
            "completion_tokens": 14,
            "seed": seed,
        }


def row(eid="ep-0000"):
    return {
        "episode_id": eid,
        "seed": 5,
        "task_digest": s.digest(TASK),
        "task": TASK,
        "main_opportunity": 1,
        "q": 0.25,
    }


def key_for(arm):
    return next(
        i.to_bytes(32, "big")
        for i in range(256)
        if s.assign(i.to_bytes(32, "big"), "p", "ep-0000")["arm"] == arm
    )


class SingleArmTests(unittest.TestCase):
    def test_wal_before_action_and_silent_executes_original_without_redraft(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep-0000"
            actor = Script([xml("DESC", "SUBMIT")])
            original = s.advance

            def check(*args):
                receipt = s.read(path / "001/assignment.json")
                self.assertEqual(
                    receipt["selection_digest"],
                    s.digest(s.read(path / "001/selection.json")),
                )
                self.assertEqual(receipt["arm"], "silent")
                return original(*args)

            with patch.object(s, "advance", side_effect=check):
                out = s.Episode(
                    path,
                    PLAN,
                    "p",
                    "r",
                    row(),
                    actor,
                    s.xml_parser(),
                    CARDS,
                    key_for("silent"),
                ).run()
            self.assertEqual(out["usage"]["actor_calls"], 1)
            self.assertEqual(out["usage"]["revision_calls"], 0)
            self.assertEqual(out["verifier"]["reward"], 1)
            self.assertFalse((path / "001/revision-request.json").exists())
            self.assertEqual(
                s.read(path / "001/execution.json")["executed_action"],
                s.read(path / "001/draft.json")["a0"],
            )
            self.assertEqual(
                out["usage"]["observed_utility"], 1 - out["usage"]["cost_penalty"]
            )

    def test_parse_failure_does_not_consume_opportunity_and_suffix_forgets_discarded_action(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep-0000"
            bad = xml("POSITIVE", "SUBMIT")
            actor = Script(["malformed response", bad, xml("DESC"), xml("SUBMIT")])
            out = s.Episode(
                path,
                PLAN,
                "p",
                "r",
                row(),
                actor,
                s.xml_parser(),
                CARDS,
                key_for("remind"),
            ).run()
            self.assertEqual(out["checkpoint_id"], "ep-0000/002")
            self.assertEqual(out["usage"]["actor_calls"], 4)
            self.assertEqual(out["usage"]["total_tokens"], 4 * 34)
            self.assertEqual(out["usage"]["revision_calls"], 1)
            self.assertEqual(out["verifier"]["reward"], 1)
            self.assertIn("Advisory note", actor.calls[2][-1]["content"])
            self.assertNotIn("Advisory note", str(actor.calls[3]))
            self.assertNotIn(bad, [m["content"] for m in actor.calls[3]])
            self.assertEqual(
                s.read(path / "002/execution.json")["executed_action"]["commands"][0][
                    "keystrokes"
                ],
                "DESC",
            )

    def test_unfunded_or_error_never_invents_other_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            actor = Script([xml("DESC", "SUBMIT")])
            out = s.Episode(
                Path(tmp) / "no",
                {**PLAN, "max_steps": 1},
                "p",
                "r",
                row(),
                actor,
                s.xml_parser(),
                CARDS,
                key_for("silent"),
            ).run()
            self.assertEqual(out["main_status"], "insufficient_revision_budget")
            self.assertIsNone(out["assignment"])
            err = s.Episode(
                Path(tmp) / "err",
                PLAN,
                "p",
                "r",
                row(),
                Script([xml("DESC", "SUBMIT"), RuntimeError()]),
                s.xml_parser(),
                CARDS,
                key_for("remind"),
            ).run()
            self.assertEqual(err["assigned_arm"], "remind")
            self.assertIsNone(err["verifier"]["reward"])
            self.assertIsNone(err["usage"]["observed_utility"])
            self.assertFalse((Path(tmp) / "err/001/execution.json").exists())

    def test_resume_preserves_partial_assignment_without_rerunning_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cohort"
            root.mkdir()
            opts = SimpleNamespace(
                root=root, parent_bank=None, source_package=None, resume=False
            )
            rows = [row(), row("ep-0001")]
            actor = Script([xml("DESC", "SUBMIT")] * 6)
            with (
                patch.object(s, "validate", return_value=(PLAN, "p", rows, CARDS)),
                patch.object(s, "FrozenQwen", return_value=actor),
                patch.object(s.secrets, "token_bytes", return_value=key_for("silent")),
            ):
                with patch.object(s, "advance", side_effect=KeyboardInterrupt()):
                    with self.assertRaises(KeyboardInterrupt):
                        s.collect(opts)
                self.assertFalse(
                    (root / "single_arm/randomization-reveal.json").exists()
                )
                opts.resume = True
                s.collect(opts)
                calls = len(actor.calls)
                s.collect(opts)
                self.assertEqual(len(actor.calls), calls)
            run = root / "single_arm"
            partial = s.read(run / "ep-0000/summary.json")
            self.assertEqual(
                (partial["termination"], partial["assigned_arm"]),
                ("interrupted", "silent"),
            )
            self.assertIsNone(partial["usage"]["observed_utility"])
            self.assertFalse((run / "ep-0000/001/execution.json").exists())
            reveal = s.read(run / "randomization-reveal.json")
            secret = bytes.fromhex(reveal["secret_hex"])
            self.assertEqual(
                hashlib.sha256(secret).hexdigest(),
                s.read(run / "manifest.json")["commitment_sha256"],
            )
            for r in rows:
                receipt = s.read(run / r["episode_id"] / "001/assignment.json")
                self.assertEqual(
                    receipt["hmac_sha256"],
                    s.assign(secret, "p", r["episode_id"])["hmac_sha256"],
                )
            self.assertEqual(s.read(run / "seal.json")["episodes"], 2)


if __name__ == "__main__":
    unittest.main()
