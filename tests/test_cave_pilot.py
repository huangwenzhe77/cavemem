import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memory_agent.cave.models import FrozenQwen
from memory_agent.cave.development_card import SOURCE_INSTRUCTION, applicable, exposure
from memory_agent.cave.pilot import check_run, script_exposed, task_plan
from memory_agent.cave.pilot_report import CELLS, DELTA_KEYS, design_intervals, outcome_cell, summarize
from memory_agent.cave.records import digest, read
from memory_agent.cave.replay import load_run, replay_tape


class FakeQwen:
    info = {"fixture": True}
    prompt = staticmethod(FrozenQwen.prompt)

    def generate(self, messages, seed):
        text = (
            "<response><analysis></analysis><plan></plan><commands>"
            '<keystrokes duration="0">SUBMIT</keystrokes></commands></response>'
        )
        return {
            "text": text,
            "seed": seed,
            "status": "ok",
            "seconds": 0.001,
            "prompt_tokens": 123,
            "completion_tokens": 32,
            "prompt_digest": digest(self.prompt(messages)),
        }


def no_hit(eid, split):
    return {
        "episode_id": eid,
        "stratum": split,
        "reached": False,
        "match": {"status": "never_reached"},
        "pairs": [],
        "unpaired_reasons": [],
        "complete_pairs": 0,
        "tape_aa": "pass",
        "free_aa": "not_requested",
        "bare_score": {"reward": 1},
        "mean_paired_delta": dict.fromkeys(DELTA_KEYS),
        "scheduled_delta_vs_bare": dict.fromkeys(
            ("success", "total_tokens", "actor_calls", "tool_calls", "accounted_seconds", "utility"), 0
        ),
        "shadow": {
            "match_seconds": 0.0,
            "bare_actor_seconds": 0.1,
            "bare_actor_calls": 1,
            "bare_input_tokens": 10,
            "bare_output_tokens": 3,
            "bare_tool_calls": 2,
            "shadow_additional_actor_calls": 0,
            "shadow_additional_tokens": 0,
            "shadow_additional_tool_calls": 0,
            "retrieval_calls": 0,
        },
    }


class PilotTests(unittest.TestCase):
    def test_descending_card_requires_all_original_records(self):
        spec = {"family": "descending", "instruction": SOURCE_INSTRUCTION, "input": [-1, -1, 3]}
        intact = {"input": spec["input"], "working": [3, -1, -1], "submitted": None}
        self.assertTrue(applicable(spec, intact))
        self.assertFalse(applicable(spec, {**intact, "working": [3, -1]}))
        self.assertFalse(applicable({**spec, "instruction": "Keep positive values only."}, intact))
        self.assertFalse(applicable(spec, {**intact, "submitted": [3, -1, -1]}))
        self.assertEqual(
            exposure({"commands": [{"keystrokes": x} for x in ("DESC", "POSITIVE", "SUBMIT")]}), 1
        )
        self.assertEqual(exposure({"commands": [{"keystrokes": x} for x in ("DESC", "SUBMIT")]}), 0)
        self.assertIsNone(exposure({"commands": [{"keystrokes": "SHOW"}]}))

    def test_forced_action_is_distinct_from_real_model_spend(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "forced"
            with patch("memory_agent.cave.pilot.FrozenQwen", return_value=FakeQwen()):
                script_exposed(SimpleNamespace(out=out, model="fixture", seed=83000))
            manifest, rows = load_run(out)
            gen = read(out / "ep-0000/002/generation.json")
            self.assertEqual(gen["origin"], "scripted_exposure_override")
            self.assertEqual(gen["prompt_tokens"], gen["shadow_actor_response"]["prompt_tokens"])
            self.assertEqual(gen["shadow_actor_action"]["commands"][0]["keystrokes"], "SUBMIT")
            self.assertEqual(
                read(out / "ep-0000/002/execution.json")["executed_action"]["commands"][0]["keystrokes"],
                "POSITIVE",
            )
            self.assertEqual(replay_tape(out, manifest, rows[0], start=2)["score"]["status"], "fail")
            plan = {
                "cohorts": [
                    {
                        "split": "controlled_transfer",
                        "seed": 83000,
                        "episodes": 1,
                        "tasks": task_plan("controlled_transfer", 83000, 1, 4),
                    }
                ]
            }
            with self.assertRaisesRegex(ValueError, "scripted"):
                check_run(plan, "controlled_transfer", out)

    def test_no_pairs_is_missing_effect_with_complete_baseline_denominator(self):
        group = [no_hit(0, "A"), no_hit(1, "A")]
        report = summarize(group)
        self.assertEqual(report["bare_pass_at_1_observed"]["rate"], 1)
        self.assertEqual(report["four_cells"], dict.fromkeys(CELLS, 0))
        self.assertEqual(report["scheduled_delta_vs_bare"]["utility"]["mean"], 0)
        self.assertIsNone(report["paired_delta_by_episode"]["utility"]["mean"])
        self.assertIsNone(report["conditional_cell_rates"]["rescue"]["rate"])
        intervals = design_intervals(
            group + [no_hit(0, "B"), no_hit(1, "B")], {"A": report, "B": report}
        )
        self.assertGreater(intervals["binary_approximate95"]["applicable"][1], 0)

    def test_four_cells_and_double_success_costs(self):
        group = []
        for i, (s, r) in enumerate([(0, 1), (1, 0), (1, 1), (0, 0)]):
            row = no_hit(i, "A")
            changes = dict.fromkeys(DELTA_KEYS, 1)
            changes.update(success=r - s, cost_usd=None)
            row.update(
                reached=True,
                match={"status": "exposed", "memory_id": "m", "rho": 1},
                free_aa="agree",
                complete_pairs=1,
                mean_paired_delta=changes,
                pairs=[
                    {
                        "cell": outcome_cell(s, r),
                        "delta": changes,
                        "arms": {"silent": {"success": s}, "remind": {"success": r}},
                    }
                ],
            )
            group.append(row)
        report = summarize(group)
        self.assertEqual(report["four_cells"], dict.fromkeys(CELLS, 1))
        self.assertEqual(report["both_pass_more_tokens"]["rate"], 1)
        self.assertEqual(report["both_pass_slower"]["rate"], 1)
        self.assertEqual(report["paired_delta_by_episode"]["success"]["mean"], 0)
        self.assertIsNone(report["paired_delta_by_episode"]["cost_usd"]["mean"])
