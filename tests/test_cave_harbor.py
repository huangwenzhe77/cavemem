"""Native Terminus2 loop with a scripted chat and in-memory terminal (no benchmark)."""
import importlib.metadata
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from memory_agent.cave.records import read

try:
    importlib.metadata.version("harbor")
except importlib.metadata.PackageNotFoundError:
    HARBOR = False
else:
    HARBOR = True


@unittest.skipUnless(HARBOR, "requires the PMA Harbor Python environment")
class HarborCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_loop_caches_exact_batch_before_dispatch(self):
        from harbor.llms.base import LLMResponse
        from harbor.llms.chat import Chat
        from harbor.models.agent.context import AgentContext
        from memory_agent.cave.harbor_capture import CapturedTerminus2
        reply = ('<response><analysis></analysis><plan></plan><commands>'
                 '<keystrokes duration="0">SHOW</keystrokes></commands></response>')
        llm = SimpleNamespace(call=AsyncMock(side_effect=[LLMResponse("malformed"), LLMResponse(reply)]))
        with tempfile.TemporaryDirectory() as tmp:
            agent = CapturedTerminus2(logs_dir=Path(tmp), max_turns=2, capture_seed=1,
                                      model_name="fixed-fixture", llm=llm, parser_name="xml",
                                      enable_summarize=False, record_terminal_session=False)
            sent = []

            async def receive(keys, **kwargs):
                draft = read(agent.capture_root / "002/draft.json")
                self.assertEqual(keys, draft["a0"]["commands"][len(sent)]["keystrokes"])
                sent.append(keys)

            agent._session = SimpleNamespace(is_session_alive=AsyncMock(return_value=True),
                                             send_keys=receive,
                                             get_incremental_output=AsyncMock(return_value="observation"))
            agent._context = AgentContext()
            # Omit the unrelated ATIF writer; use the real model loop, parser,
            # Chat bookkeeping, step construction, send_keys and capture hooks.
            with patch.object(agent, "_dump_trajectory"):
                await agent._run_agent_loop("fixture", Chat(llm))
            self.assertEqual(sent, ["SHOW"])
            index = read(agent.capture_root / "index.json")
            self.assertEqual([s["opportunity_index"] for s in index["steps"]], [None, 1])
            execution = read(agent.capture_root / "002/execution.json")
            self.assertEqual(execution["executed_action"], read(agent.capture_root / "002/draft.json")["a0"])

    async def test_upload_failure_is_not_a_task_failure(self):
        from memory_agent.cave.harbor_capture import verify_harbor
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "test.sh").write_text("exit 0")
            env = SimpleNamespace(
                exec=AsyncMock(side_effect=[SimpleNamespace(return_code=0),
                               SimpleNamespace(return_code=1, stdout=None)]),
                upload_dir=AsyncMock(),
            )
            verdict = await verify_harbor(env, SimpleNamespace(tests_dir=tmp, max_test_timeout_sec=20))
            self.assertEqual((verdict["status"], verdict["reward"]), ("unavailable", None))

    async def test_zero_reward_without_promised_test_report_is_unavailable(self):
        from memory_agent.cave.harbor_capture import verify_harbor
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "test.sh"
            script.write_text("pytest --ctrf /logs/verifier/ctrf.json", encoding="utf-8")
            checksum = hashlib.sha256(script.read_bytes()).hexdigest()
            env = SimpleNamespace(upload_dir=AsyncMock(), exec=AsyncMock(side_effect=[
                SimpleNamespace(return_code=0), SimpleNamespace(return_code=0, stdout=checksum),
                SimpleNamespace(return_code=0), SimpleNamespace(return_code=0, stdout="0"),
                SimpleNamespace(return_code=1, stdout=None)]))
            verdict = await verify_harbor(env, SimpleNamespace(tests_dir=tmp, max_test_timeout_sec=20))
            self.assertEqual((verdict["status"], verdict["reward"]), ("unavailable", None))
