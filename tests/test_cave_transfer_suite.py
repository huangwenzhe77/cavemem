import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from memory_agent.cave.transfer_suite import local_check, query_note


class TransferTests(unittest.TestCase):
    def test_guarded_live_decision_controls_the_only_execution(self):
        from memory_agent.cave import transfer_suite as suite
        from test_cave_single_arm import CARDS, PLAN, Script, row, xml
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"episode"
            actor=Script([xml("DESC","SUBMIT")])
            ep=suite.LiveEpisode(dest,PLAN,"plan","run",row(),actor,suite.xml_parser(),CARDS,None)
            ep.system=suite.GUARDED
            ep.dr_policy=MagicMock()
            ep.dr_policy.decide.return_value={"remind":False,"reason":"fixture_abstain","probability":0.}
            out=ep.run()
            self.assertEqual(out["verifier"]["reward"],1)
            self.assertEqual(out["usage"]["actor_calls"],1)
            self.assertEqual(out["assignment"]["reason"],"fixture_abstain")
            self.assertEqual(ep.dr_policy.decide.call_args.kwargs,{"head":"dr","guarded":True})

    def test_checker_protects_multiplicity_and_leaves_real_state_alone(self):
        t = {"input":[2,-1,2], "family":"ascending"}
        s = {"input":list(t["input"]), "working":list(t["input"]), "submitted":None}
        before = deepcopy(s)
        for ops, trigger in ((["SHOW"],False), (["UNIQUE"],True), (["ASC","SUBMIT"],False), (["SUBMIT"],True)):
            a = {"commands":[{"keystrokes":o} for o in ops],"complete":False}
            self.assertEqual(local_check(t,s,a),trigger)
            self.assertEqual(s,before)
        self.assertIn("3 entries",query_note("retain all",t,s["working"]))


class HarborTransferTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_pma_path_uses_post_run_grader_and_retains_memory_stats(self):
        try:
            from memory_agent import runner
        except ModuleNotFoundError:
            self.skipTest("requires PMA/Harbor environment")
        from memory_agent.config import load_config
        from memory_agent.data.base import Task
        cfg = load_config(Path("configs/cave_harbor.yaml"))
        cfg.run.cave_capture = False
        cfg.run.harbor_trusted_grading = True
        cfg.memory.enabled = True
        cfg.memory.save_memory_per_task = False
        task = Task(task_name="regex-log",instruction="write a regex",test_weights={})
        env = MagicMock()
        env._container_name = None
        env.start,env.stop = AsyncMock(), AsyncMock()
        env.exec = AsyncMock(return_value=SimpleNamespace(return_code=0,stdout=""))
        agent = MagicMock()
        agent.run,agent.setup = AsyncMock(),AsyncMock()
        agent.get_memory_stats.return_value={"memory_size":1,"injection_count":2}
        verdict = {"status":"unavailable","reward":None,"reason":"missing_grader"}
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner, "EnrootEnvironment", return_value=env),
            patch.object(runner, "create_agent", return_value=agent),
            patch("memory_agent.cave.harbor_capture.verify_harbor", new=AsyncMock(return_value=verdict)) as grade,
        ):
            result = await runner.run_task(task,cfg,Path(tmp))
            grade.assert_awaited_once_with(env,task)
            self.assertIsNone(result.reward)
            self.assertEqual(result.error,"missing_grader")
            self.assertEqual(result.memory_stats["injection_count"],2)
            self.assertFalse((Path(tmp)/"regex-log/verifier/reward.txt").exists())
