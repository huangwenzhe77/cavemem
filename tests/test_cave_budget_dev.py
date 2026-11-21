import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from memory_agent.cave import runtime_pair
from memory_agent.cave.budget_fork import fork
from test_cave_single_arm import CARDS, PLAN, Script, row, xml


class RemainingCallBudgetTests(unittest.TestCase):
    def test_common_turn_horizon_and_one_remaining_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            t=row()
            mf={"actor":PLAN["actor"],"max_steps":4,"run_id":"fixture","split":"development","cohort":"natural"}
            policy=MagicMock()
            policy.base.model={"cards":{"fixture":"descending"}}
            policy.decide.return_value={"remind":False}
            sample=runtime_pair.episode(run,t,mf,PLAN,Script([xml("SHOW"),xml("DESC","SUBMIT")]),
                                        runtime_pair.xml_parser(),CARDS,policy)
            results=[]
            for cap in (3,1):
                actor=Script([xml("SHOW"),xml("DESC","SUBMIT")])
                out=fork(run,mf,sample,1,actor,"remind","Keep the numbers.",139,16384,64,cap)
                results.append(out)
            rich,limited=results
            self.assertEqual(rich["limits"]["turns"],limited["limits"]["turns"])
            self.assertEqual(rich["events"][0]["executed"],limited["events"][0]["executed"])
            self.assertEqual(rich["score"]["reward"],1)
            self.assertEqual(limited["termination"],"actor_budget")
            self.assertEqual(limited["score"]["reward"],0)
            self.assertEqual(limited["usage"]["actor_calls"]-limited["common_prefix"]["actor_calls"],1)
