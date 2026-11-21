import unittest

import numpy as np

from memory_agent.cave.policy_value import POLICIES, dr_scores, evaluate


class PolicyValueTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.pred = [], []
        for fold in ("a", "b", "c"):
            for z in (0, 1):
                self.rows.append({"z": z, "p_logged": .5, "r": z, "y": z, "usage": {"total_tokens": 100*z},
                                  "cluster_id": fold+str(z), "assigned_arm": ("silent", "remind")[z]})
                self.pred.append({"fold": fold, "ready": False, "policy_seconds": dict.fromkeys(POLICIES, 0.),
                                  "actions": {p: int(p != "silent") for p in POLICIES}})

    def test_deltas_and_unchanged_tasks(self):
        common = {**self.rows[0], "cluster_id": "other", "assigned_arm": None}
        report = evaluate(self.rows, self.rows+[common], self.pred, .01, {"policy_mean_extra_tokens_budget": 256})
        remind = report["always_remind"]
        self.assertEqual(remind["ess"], 3)
        self.assertEqual(remind["eligible_delta_vs_silent"]["tokens"], 100)
        self.assertAlmostEqual(remind["all_delta_vs_silent"]["utility"], 6/7)
        self.assertEqual(remind["constraints"]["observed_cohort_possible_regressions"], [0, 0])

    def test_held_out_outcome_never_enters_its_outcome_model(self):
        before, _ = dr_scores(self.rows, self.pred, 0.)
        self.rows[0]["r"] += 5
        after, _ = dr_scores(self.rows, self.pred, 0.)
        self.assertEqual(after[0, 0, 0] - before[0, 0, 0], 10)
        np.testing.assert_array_equal(before[1], after[1])

    def test_missing_support_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "support"):
            dr_scores(self.rows[::2], self.pred[::2], 0.)
