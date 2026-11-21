import tempfile
import unittest
from pathlib import Path

from memory_agent.cave import gate, gate_eval
from memory_agent.cave.records import digest, save


CARDS = {"desc": "descending"}


def make_x(ops):
    return {
        "memory": {"id": "desc"},
        "instruction": "Return every value in decreasing order.",
        "working_state": {
            "working": [-1, 2, 2],
            "input": [-1, 2, 2],
            "submitted": None,
        },
        "proposal": {"commands": [{"keystrokes": c} for c in ops], "complete": False},
        "budget_after_proposal": {"remaining_actor_calls": 2},
        "evidence": {"source_verified": 1, "applicable": 1, "exposed": 0},
    }


class GateTests(unittest.TestCase):
    def test_symbolic_submission_checks_repeats_signs_and_last_submission(self):
        self.assertTrue(gate.ready(make_x(["DESC", "SUBMIT", "POSITIVE"]), CARDS))
        self.assertFalse(
            gate.ready(make_x(["DESC", "SUBMIT", "UNIQUE", "SUBMIT"]), CARDS)
        )
        self.assertFalse(gate.ready(make_x(["POSITIVE", "DESC", "SUBMIT"]), CARDS))
        self.assertFalse(gate.ready(make_x(["DESC"]), CARDS))

    def test_ablation_ignores_all_current_action_fields_and_postdecision_values(self):
        x = make_x(["DESC", "SUBMIT"])
        before = gate.features_for(x, CARDS, "no_a0")
        full = gate.features_for(x, CARDS, "full")
        del x["proposal"]
        del x["evidence"]
        x.update(
            parse_warning="changed",
            history_before_proposal=["outcome"],
            targets={"H": 1},
        )
        x["budget_after_proposal"]["remaining_tokens"] = -999
        self.assertEqual(before, gate.features_for(x, CARDS, "no_a0"))
        self.assertEqual(set(before), set(gate.BASE))
        x = make_x(["DESC", "SUBMIT"])
        x.update(targets={"H": 1}, future="pass", hard_label="remind")
        self.assertEqual(full, gate.features_for(x, CARDS, "full"))

    def test_weight_per_task_and_missing_risk_class(self):
        rows = [
            {
                "sample": {
                    "features": make_x(["SHOW"]),
                    "targets": {"D": value, "H": 0},
                },
                "task_id": task,
            }
            for task, value in [("a", 1), ("a", 1), ("b", 0)]
        ]
        variants, w = gate.weighted_fit(rows, CARDS)
        self.assertEqual(w, [0.5, 0.5, 1])
        self.assertEqual(variants["full"]["D"]["means"], [0.5])
        self.assertIsNone(variants["full"]["H"])
        m = {"cards": CARDS, "variants": variants, "threshold": 0.5}
        self.assertFalse(gate.predict(m, rows[0]["sample"]["features"])["remind"])

    def test_runtime_cost_and_model_signature(self):
        plan = {"lambda_tokens": 0.01, "lambda_tool": 0.001, "lambda_time": 0.01}
        usage = {
            "input_tokens": 60,
            "output_tokens": 40,
            "total_tokens": 100,
            "tool_calls": 1,
            "accounted_seconds": 60,
            "actor_calls": 1,
            "success": 1,
        }
        a = gate_eval.score(usage, 6, plan)
        self.assertEqual(a["accounted_seconds"], 66)
        self.assertAlmostEqual(a["utility"], 1 - 0.001 - 0.001 - 0.011)
        self.assertIsNone(gate_eval.score(None, 6, plan))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.json"
            body = {"threshold": 0.002}
            save(p, {"model_id": digest(body), "threshold": -0.01})
            with self.assertRaises(ValueError):
                gate.signed(p, "model_id")


if __name__ == "__main__":
    unittest.main()
