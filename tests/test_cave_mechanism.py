import unittest

from memory_agent.cave.mechanism_selector import terminal_or_budget_boundary


class MechanismSelectorTests(unittest.TestCase):
    def classify(self, commands, turn=1, working=(1, 3, 3), warning=False, complete=False):
        task = {"family": "descending", "input": [1, 3, 3]}
        state = {"submitted": None, "working": list(working)}
        pre = {"step": turn, "state": state}
        draft = {"error": warning, "a0": {"complete": complete, "commands": [
            {"keystrokes": op} for op in commands
        ]}}
        return terminal_or_budget_boundary(task, pre, draft, {"max_steps": 4})

    def test_submit_requires_preserved_records_and_the_requested_order(self):
        self.assertEqual(self.classify(["DESC", "SUBMIT"]), "ready_to_submit")
        self.assertEqual(self.classify(["SUBMIT"], working=(3, 3, 1)), "ready_to_submit")
        for ops in (["SUBMIT"], ["ASC", "SUBMIT"], ["POSITIVE", "DESC", "SUBMIT"],
                    ["SUBMIT", "DESC"], ["DESC", "SUBMIT", "SUBMIT"]):
            self.assertIsNone(self.classify(ops))
        self.assertIsNone(self.classify(["DESC", "SUBMIT"], working=(3, 1)))
        self.assertIsNone(self.classify(["DESC", "SUBMIT"], warning=True))

    def test_budget_boundary_gives_the_original_one_real_call(self):
        self.assertEqual(self.classify(["DESC"], turn=3), "one_call_left_to_submit")
        self.assertEqual(self.classify(["SHOW"], turn=3, working=(3, 3, 1)), "one_call_left_to_submit")
        self.assertIsNone(self.classify(["SHOW"], turn=3))
        self.assertIsNone(self.classify(["DESC"], turn=2))
        self.assertIsNone(self.classify(["DESC"], turn=3, complete=True))
        self.assertIsNone(self.classify(["DESC", "SUBMIT"], turn=4))
