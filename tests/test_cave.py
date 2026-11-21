import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from io import BytesIO

from memory_agent.cave import mini
from memory_agent.cave.__main__ import collect_episode, propose
from memory_agent.cave.models import TeacherCallError, teacher_request, xml_parser
from memory_agent.cave.records import read, save, verify


def xml(*ops):
    commands = "".join(f'<keystrokes duration="0">{op}</keystrokes>' for op in ops)
    return f"<response><analysis></analysis><plan></plan><commands>{commands}</commands></response>"


class Script:
    def __init__(self, *responses):
        self.responses = iter(responses)

    def generate(self, messages, seed):
        text = next(self.responses)
        return {"text": text, "prompt_tokens": 8, "completion_tokens": 6, "seed": seed}


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.task = mini.task("source", 2)

    def test_parse_error_does_not_consume_opportunity(self):
        import random
        seed = next(n for n in range(100) if random.Random("checkpoint:"+str(n)).randint(1, 3) == 1)
        actor = Script("malformed", xml("ASC", "SUBMIT"))
        item = collect_episode(self.root, actor, xml_parser(), self.task, "ep", seed, 3)
        self.assertEqual([s["opportunity_index"] for s in item["steps"]], [None, 1])
        self.assertEqual(item["main_turn_id"], 2)
        self.assertEqual(item["verifier"]["status"], "pass")
        folder = self.root / "ep/002"
        pre, draft, execution = [read(folder / f"{name}.json") for name in ("pre", "draft", "execution")]
        self.assertEqual(pre["state"], draft["state"])
        self.assertEqual(draft["a0"], execution["executed_action"])
        self.assertEqual(pre["messages"], draft["messages"][:-1])
        # Restore pre-execution state and replay the entire batch including SUBMIT.
        state = draft["state"]
        self.assertEqual(mini.execute(state, draft["a0"]["commands"]), execution["observation"])
        self.assertEqual(state, execution["state"])

    def test_model_error_and_unreached_schedule_stay_visible(self):
        item = collect_episode(self.root, Script(), xml_parser(), self.task, "ep", 5, 1)
        self.assertEqual(item["steps"][0]["status"], "actor_error")
        self.assertEqual(item["main_status"], "never_reached")
        self.assertIsNone(item["verifier"]["reward"])

    def test_task_has_no_host_operations(self):
        state = mini.initial(self.task)
        path = self.root / "sentinel"
        obs = mini.execute(state, [{"keystrokes": "touch "+str(path)}])
        self.assertTrue(obs[0].startswith("rejected"))
        self.assertFalse(path.exists())
        self.assertEqual(state, mini.initial(self.task))

    def test_verdicts_and_stale_id(self):
        state = mini.initial(self.task)
        failure = verify(self.task, state)
        self.assertEqual(failure["status"], "fail")
        state["submitted"] = sorted(self.task["input"])
        success = verify(self.task, state)
        self.assertEqual(success["status"], "pass")
        self.assertNotEqual(failure["request_id"], success["request_id"])
        self.assertIsNone(verify(self.task, state, self.root / "absent")["reward"])
        stale = self.root / "stale.py"
        stale.write_text('print(\'{"request_id":"old","reward":1}\')', encoding="utf-8")
        self.assertEqual(verify(self.task, state, stale)["status"], "unavailable")

    def test_teacher_cannot_open_target_trace(self):
        save(self.root / "manifest.json", {"split": "target_test"})
        with patch("memory_agent.cave.__main__.teacher_request") as request:
            with self.assertRaisesRegex(ValueError, "source"):
                propose(SimpleNamespace(run=self.root))
            request.assert_not_called()

    def test_teacher_reports_http_metadata_and_rejects_model_substitution(self):
        source = {"steps": [{"step": 1}]}
        for returned_model in ("gpt-6-sol", "different"):
            candidate = {"step": 1, "mechanism": "order", "alternative": ["ASC"],
                         "applies_when": "numeric ascending requested", "excludes_when": "descending",
                         "reminder": "Check sorting direction."}
            body = {"model": returned_model, "choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps(candidate)}}]}
            reply = BytesIO(json.dumps(body).encode())
            reply.status = 200
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fixture-value"}), patch(
                    "urllib.request.urlopen", return_value=reply) as transport:
                if returned_model == "gpt-6-sol":
                    result = teacher_request(source, "https://example.org", returned_model)
                    self.assertEqual(result["http_status"], 200)
                    self.assertIsNone(result["usage"])
                    self.assertNotIn("fixture-value", json.dumps(result))
                else:
                    with self.assertRaises(TeacherCallError) as raised:
                        teacher_request(source, "https://example.org", "gpt-6-sol")
                    self.assertEqual(raised.exception.audit["response_model"], "different")
                self.assertEqual(transport.call_args.args[0].full_url, "https://example.org/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
