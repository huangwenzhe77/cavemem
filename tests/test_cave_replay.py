import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from memory_agent.cave import mini
from memory_agent.cave.__main__ import collect_episode
from memory_agent.cave.branches import branch
from memory_agent.cave.experiment import audit_aa, match
from memory_agent.cave.models import FrozenQwen, xml_parser
from memory_agent.cave.records import digest, read, save
from memory_agent.cave.replay import replay_tape, restore


def xml(*ops):
    return (
        "<response><analysis></analysis><plan></plan><commands>"
        + "".join(f'<keystrokes duration="0">{op}</keystrokes>' for op in ops)
        + "</commands></response>"
    )


class FixtureActor:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.inputs = []

    def count_prompt_tokens(self, messages):
        return 12

    def generate(self, messages, seed):
        self.inputs.append(json.loads(json.dumps(messages)))
        return {
            "text": next(self.responses),
            "seed": seed,
            "status": "ok",
            "seconds": 0.001,
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "prompt_digest": digest(FrozenQwen.prompt(messages)),
        }


class ReplayTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.spec = mini.task("controlled_transfer", 321)
        self.manifest = {
            "run_id": "fixture",
            "split": "controlled_transfer",
            "tasks": [self.spec],
            "benchmark": "mini-data-v1",
            "max_steps": 4,
            "memory_enabled": False,
            "teacher_enabled": False,
            "actor": {"max_new_tokens": 64, "do_sample": False, "temperature": 0},
        }
        save(self.run / "manifest.json", self.manifest)
        actor = FixtureActor([xml("ASC"), xml("POSITIVE"), xml("SHOW"), xml("SUBMIT")])
        self.item = collect_episode(self.run, actor, xml_parser(), self.spec, "ep-0000", 321, 4)
        self.item.update(run_id="fixture", split="controlled_transfer")
        (self.run / "index.jsonl").write_text(json.dumps(self.item) + "\n", encoding="utf-8")

    def test_independent_state_and_corruption_detection(self):
        a = restore(self.run, self.item, 2)
        b = restore(self.run, self.item, 2)
        a[0]["state"]["working"].clear()
        self.assertTrue(b[0]["state"]["working"])
        tape = replay_tape(self.run, self.manifest, self.item, start=2)
        self.assertEqual(tape["score"]["status"], "fail")
        broken = self.run / "ep-0000/002/pre.json"
        save(broken, a[0])
        with self.assertRaisesRegex(ValueError, "Hpre"):
            replay_tape(self.run, self.manifest, self.item, start=2)
        out = self.root / "aa"
        audit_aa(SimpleNamespace(run=self.run, out=out, free=False))
        self.assertEqual(read(out / "aa.json")["checks"][0]["tape_status"], "failed")

    def test_live_revision_is_once_and_never_sees_the_other_draft(self):
        reminder = "Keep all records."
        silent = branch(
            self.run,
            self.manifest,
            self.item,
            2,
            FixtureActor([xml("SHOW"), xml("SUBMIT")]),
            "silent",
            reminder,
            1,
            4096,
            32,
        )
        actor = FixtureActor([xml("SHOW"), xml("SUBMIT")])
        remind = branch(self.run, self.manifest, self.item, 2, actor, "remind", reminder, 1, 4096, 32)
        self.assertEqual((silent["revision_calls"], remind["revision_calls"]), (0, 1))
        self.assertEqual(silent["events"][0]["executed"]["commands"][0]["keystrokes"], "POSITIVE")
        self.assertIn(reminder, str(actor.inputs[0]))
        self.assertNotIn(reminder, str(actor.inputs[1]))
        self.assertNotIn(xml("POSITIVE"), str(actor.inputs[0]))
        self.assertNotIn(xml("POSITIVE"), str(actor.inputs[1]))
        self.assertEqual((silent["score"]["reward"], remind["score"]["reward"]), (0, 1))
        self.assertLessEqual(remind["usage"]["actor_calls"], remind["limits"]["actor_calls"])
        limited = branch(
            self.run,
            self.manifest,
            self.item,
            2,
            FixtureActor([xml("SHOW"), xml("SHOW")]),
            "remind",
            reminder,
            1,
            4096,
            32,
        )
        self.assertEqual((limited["termination"], limited["usage"]["actor_calls"]), ("actor_budget", 4))

    def test_semantics_and_exposure_are_independent(self):
        pre, draft, _ = restore(self.run, self.item, 2)
        cards = [{"memory_id": "fixed", "candidate": {"reminder": "Keep all records."}}]
        for op, status, exposure in [
            ("POSITIVE", "exposed", 1),
            ("SUBMIT", "avoided", 0),
            ("SHOW", "unknown_exposure", None),
        ]:
            draft["a0"]["commands"][0]["keystrokes"] = op
            decision = match(cards, self.spec, pre, draft)
            self.assertEqual((decision["status"], decision["rho"], decision["e"]), (status, 1, exposure))
        self.assertEqual(
            match(cards, mini.task("target_train", 321), pre, draft)["status"], "no_candidate"
        )

    def test_terminal_source_contrast_and_nonterminal_rejection(self):
        fixed = {"commands": [{"keystrokes": "SUBMIT", "duration": 0.0}], "complete": False}
        repaired = replay_tape(self.run, self.manifest, self.item, start=2, replacement=fixed)
        self.assertEqual(repaired["score"]["reward"], 1)
        fixed["commands"][0]["keystrokes"] = "SHOW"
        with self.assertRaisesRegex(ValueError, "nonterminal"):
            replay_tape(self.run, self.manifest, self.item, start=2, replacement=fixed)
