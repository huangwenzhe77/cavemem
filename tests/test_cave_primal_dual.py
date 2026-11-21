import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from memory_agent.cave.auxiliary_heads import fit_heads, predict

from memory_agent.cave.constrained_gate import CONFIG, optimize
from memory_agent.cave.diagnostic_trial import diagnostic, draw
from memory_agent.cave.probe_value import adaptive_decision, estimate


class PolicyTests(unittest.TestCase):
    def test_auxiliary_heads_do_not_invent_harm_labels(self):
        X = np.tile([[0.], [1.]], (10, 1))
        g = np.arange(20)//4
        h = X[:,0]
        cost = h*30
        empty = fit_heads(X, cost, h*0, g)
        self.assertIsNone(predict(empty, X[0])["regression_probability"])
        fitted = fit_heads(X, cost, h, g)
        low, high = (predict(fitted, v) for v in X[:2])
        self.assertLess(low["regression_probability"], high["regression_probability"])
        self.assertLess(low["extra_tokens"], high["extra_tokens"])

    def test_constraints_activate_and_update_weights(self):
        # Synthetic regression and costly reminders. Never mixed into real labels.
        n = 64
        X = np.ones((n, 1))
        initial = [np.zeros((1, 2)), np.zeros(2), np.ones(2), np.zeros(1)]
        result = optimize(X, np.full(n, .01), np.ones(n), np.full(n, 800.), initial, 0.,
                          {"policy_risk_delta": .05, "policy_mean_extra_tokens_budget": 100},
                          .01, {**CONFIG, "steps": 150})
        self.assertGreater(max(r["alpha"] for r in result["trace"]), 0.)
        self.assertGreater(max(r["beta"] for r in result["trace"]), 0.)
        self.assertLess(result["trace"][-1]["mean_remind_probability"], .05)
        self.assertEqual(result["trace"][0]["KL"], 0.)
        np.testing.assert_array_equal(initial[2], np.ones(2))

    def test_probe_read_only_and_budget_gated(self):
        state = {"working": [0, 0, 1], "submitted": None}
        before = deepcopy(state)
        result = diagnostic(state, {"commands": [{"keystrokes": "UNIQUE"}]})
        self.assertEqual(state, before)
        self.assertEqual(result, {"would_remove_records": True})
        policy = SimpleNamespace(doc={"policy_id": "p"}, decide=lambda x,u: {"remind": False})
        x = {"evidence": {"applicable": 1, "source_verified": 1, "exposed": None},
             "budget_after_proposal": {"remaining_actor_calls": 2, "remaining_tools": 2}}
        report = {"policy_id": "p", "uncertain_continuation": {"post": [[1,0],[0,1]]},
                  "uncertain": {"voi_lower": .2}}
        for budget, risk in ((0,True), (1,False)):
            self.assertFalse(adaptive_decision(x,policy,report,.3,budget,risk,
                                               lambda: self.fail("probe called"))["inspected"])
        self.assertTrue(adaptive_decision(x,policy,report,.3,1,True,lambda: result)["remind"])
        self.assertNotEqual(draw("ab"*32,"e","stage1"), draw("ab"*32,"e","stage2"))

    def test_sequential_identification_and_sparse_folds(self):
        rows = []
        for fold in "ABC":
            for action in ("silent", "remind", "inspect"):
                for flag in (False, True):
                    for z in (0, 1):
                        actual = int(action=="remind") if action!="inspect" else z
                        rows.append({"first": action, "fold": fold, "r": float(actual==flag),
                                     "z": actual, "b1": 1/3, "b2": .5, "flag": flag, "base_p": .5})
        r = estimate(rows)
        self.assertAlmostEqual(r["direct_dr"], .5)
        self.assertAlmostEqual(r["inspect_dr"], 1.)
        self.assertAlmostEqual(r["net_voi_ips"], .5)
        self.assertIsNone(estimate(rows[:2])["voi_lower"])
