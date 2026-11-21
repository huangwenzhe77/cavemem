import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from memory_agent.cave import runtime_pair as sampler
from test_cave_single_arm import CARDS, PLAN, Script, row, xml


class RuntimePairTests(unittest.TestCase):
    def test_second_eligible_draft_persists_online_prediction_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            r = row()
            r["seed"] = next(i for i in range(1000) if sampler.uniform("selection", i, 1) >= .5
                             and sampler.uniform("selection", i, 2) < .5)
            model = Mock()
            model.base.model = {"cards":{"fixture":"descending"}}
            model.decide.return_value = {"remind":False}
            advance = sampler.advance
            turns = []

            def execute(*args):
                cp = run / r["episode_id"] / f"{len(turns)+1:03d}"
                turns.append(sampler.read(cp / "hazard.json"))
                self.assertEqual((cp / "prediction.json").exists(), len(turns)==2)
                return advance(*args)

            actor = Script([xml("SHOW"), xml("SHOW"), xml("DESC", "SUBMIT")])
            mf = {"actor":PLAN["actor"], "max_steps":4, "run_id":"fixture", "split":"development"}
            with patch.object(sampler, "advance", side_effect=execute):
                item = sampler.episode(run, r, mf, PLAN, actor, sampler.xml_parser(), CARDS, model)
            self.assertEqual(item["selected"], "ep-0000/002")
            self.assertEqual(item["eligible_draws"], 2)
            self.assertEqual([s["first_hit_prob"] for s in turns[:2]], [.5,.25])
            self.assertEqual(turns[2]["reason"], "after_first_hit")
            self.assertEqual(item["verifier"]["reward"], 1)

    def test_unaffordable_draft_generates_no_draw(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            plan = {**PLAN, "max_steps":1}
            mf = {**plan, "run_id":"fixture", "split":"development"}
            policy = SimpleNamespace()
            item = sampler.episode(run,row(),mf,plan,Script([xml("DESC","SUBMIT")]),
                                   sampler.xml_parser(),CARDS,policy)
            self.assertIsNone(item["selected"])
            self.assertEqual(item["eligible_draws"], 0)
            self.assertEqual(sampler.read(run/"ep-0000/001/hazard.json")["reason"], "budget")
