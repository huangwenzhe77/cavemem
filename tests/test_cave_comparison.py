import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from memory_agent.cave import factorial_study as factorial
from memory_agent.cave import logged_policy_fit as fit
from memory_agent.cave.records import digest


class ComparisonTests(unittest.TestCase):
    def test_dr_training_derivative_and_fold_isolation(self):
        rows = [{"episode_id":str(i), "block":i//2, "x":{}, "z":i%2, "b":.5,
                 "observed":{"r":float(i%2),"tokens":10*(i%2)}} for i in range(16)]
        with patch.object(fit.gate,"ready",return_value=False):
            before, folds = fit.pseudo_advantages(rows,{})
            rows[0]["observed"]["r"] += 2
            after, changed = fit.pseudo_advantages(rows,{})
        np.testing.assert_array_equal(before,np.tile([1.,10.],(16,1)))
        self.assertEqual(folds[0]["mu"],changed[0]["mu"])
        self.assertEqual(folds[1]["mu"],changed[1]["mu"])
        self.assertEqual(after[0,0]-before[0,0],-4)

    def test_confidence_fallback_and_unsupported_domain(self):
        base=SimpleNamespace(predict=lambda x:{"predicted_D":x["score"]})
        def example(i):
            return {"block":i,"x":{"score":.1,"instruction":"desc"},
                    "arms":{"silent":{"r":1.,"y":1},"remind":{"r":1.01,"y":1}}}
        cal=fit.calibrate([example(i) for i in range(18)],base,.002,.05)
        self.assertAlmostEqual(cal["offset"],.09)
        self.assertFalse(cal["risk_supported"])
        self.assertGreater(cal["risk_upper"],.05)
        self.assertLess(fit.risk_upper(0,100),.05)
        model=object.__new__(fit.Runtime)
        model.doc={"calibration":cal,"profiles":{"id":{}}}
        model.base=base
        model.params={"dr":[np.zeros((1,1)),np.zeros(1),np.ones(1),np.array([1.])]}
        x={**example(0)["x"],"memory":{"id":"id"},"evidence":{"source_verified":1,"applicable":1},
           "budget_after_proposal":{"remaining_actor_calls":1}}
        with patch.object(fit,"features",return_value=np.ones(1)):
            self.assertEqual(model.decide(x,0.)["reason"],"insufficient_policy_risk_evidence")
            self.assertTrue(model.decide(x,0.,guarded=False)["remind"])
            self.assertEqual(model.decide({**x,"instruction":"other"},0.)["reason"],"outside_calibration_domain")

    def test_four_cells_noise_and_missing_transport_support(self):
        from memory_agent.cave.models import xml_parser
        from memory_agent.cave.experiment import applies
        task={"family":"ascending","instruction":"Sort ascending","input":[3,-1,3]}
        for rho in (0,1):
            for e in (0,1):
                pre,draft,_=factorial.prefix(task,"ep",rho,e,xml_parser())
                self.assertEqual(applies(pre,task),rho)
                self.assertEqual(fit.commands(draft["a0"]),["POSITIVE"] if e else ["SUBMIT"])
        c={"candidate":{"reminder":"retain all"}}
        c["memory_id"]=digest(c)
        bank=factorial.adversarial_bank(c)
        self.assertTrue(factorial.vetted(c,{c["memory_id"]}))
        self.assertFalse(factorial.vetted(bank["contradictory"],{c["memory_id"]}))
        v=factorial.standardize({"11":[1,2,3,4]},{"11":4,"unknown":6})
        self.assertEqual(v["covered_only"],[1,2,3,4])
        self.assertIsNone(v["population"])
        self.assertEqual(v["coverage"],.4)
