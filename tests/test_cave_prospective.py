import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memory_agent.cave import mini, prospective
from memory_agent.cave.__main__ import collect_episode
from memory_agent.cave.models import FrozenQwen, xml_parser
from memory_agent.cave.pilot import task_plan
from memory_agent.cave.records import digest, read, save
from memory_agent.cave.replay import protocol_digest


class ActorFixture:
    info = {"max_new_tokens": 256, "do_sample": False, "temperature": 0}

    def count_prompt_tokens(self, messages):
        return 20

    def generate(self, messages, seed):
        revision = "Advisory note" in messages[-1]["content"]
        ops = ["DESC", "SUBMIT"] if revision else ["DESC", "POSITIVE", "SUBMIT"]
        text = (
            "<response><analysis></analysis><plan></plan><commands>"
            + "".join(f'<keystrokes duration="0">{op}</keystrokes>' for op in ops)
            + "</commands></response>"
        )
        return {
            "status": "ok",
            "seed": seed,
            "text": text,
            "seconds": 0.01,
            "prompt_tokens": 20,
            "completion_tokens": 15,
            "prompt_digest": digest(FrozenQwen.prompt(messages)),
        }


class ProspectiveTests(unittest.TestCase):
    cards = {
        "ascending": {
            "memory_id": "asc",
            "contract": "asc",
            "candidate": {"reminder": "Submit the full sorted list."},
        },
        "descending": {
            "memory_id": "desc",
            "contract": "desc",
            "candidate": {"reminder": "Keep every record."},
        },
    }

    def test_two_cards_route_by_task_then_exposure(self):
        s = mini.task("target_train", 32)
        pre = {"state": mini.initial(s), "messages": []}
        draft = {"a0": {"commands": [{"keystrokes": "SHOW"}]}}
        m = prospective.route(self.cards, s, pre, draft)
        self.assertEqual((m["memory_id"], m["rho"], m["e"]), ("desc", 1, None))
        pre["state"]["working"].pop()
        self.assertEqual(prospective.route(self.cards, s, pre, draft)["rho"], 0)
        a = mini.task("controlled_transfer", 32)
        self.assertEqual(
            prospective.route(self.cards, a, {"state": mini.initial(a), "messages": []}, draft)[
                "memory_id"
            ],
            "asc",
        )

    def test_new_entrypoint_restores_scores_and_reconciles_both_branches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actor = ActorFixture()
            entries = []
            for n, split in enumerate(("target_train", "controlled_transfer")):
                seed = next(
                    s
                    for s in range(40 * n, 40 * n + 39)
                    if random.Random("checkpoint:" + str(s)).randint(1, 4) == 1
                    and min(mini.task(split, s)["input"]) <= 0
                )
                planrows = task_plan(split, seed, 1, 4)
                run = root / split
                run.mkdir()
                row = planrows[0]
                manifest = {
                    "run_id": "fixture-" + split,
                    "split": split,
                    "benchmark": "mini-data-v1",
                    "tasks": [row["task"]],
                    "actor": actor.info,
                    "max_steps": 4,
                    "seed": seed,
                    "teacher_enabled": False,
                    "memory_enabled": False,
                }
                save(run / "manifest.json", manifest)
                result = collect_episode(run, actor, xml_parser(), row["task"], "ep-0000", seed, 4)
                result.update(run_id=manifest["run_id"], split=split)
                (run / "index.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
                entries.append({**row, "stratum": split, "task_digest": digest(row["task"])})
            plan = {
                "protocol": protocol_digest(),
                "actor": actor.info,
                "max_new_tokens": 256,
                "max_steps": 4,
                "temperature": 0,
                "router_version": prospective.ROUTER,
                "cohort_version": "prospective-v2",
                "descending_memory_id": "desc",
                "bank_id": "fixture-bank",
                "parent_bank_id": "fixture-parent",
                "replicates": 1,
                "branch_seed": 7331,
                "total_tokens": 16384,
                "total_tools": 64,
                "lambda_tokens": 0.01,
                "lambda_time": 0.01,
                "lambda_tool": 0.001,
                "tasks": entries,
            }
            save(root / "preregistration.json", plan)
            with (
                patch.object(prospective, "verified_cards", return_value=self.cards),
                patch.object(prospective, "load_actor", return_value=actor),
            ):
                for split in ("target_train", "controlled_transfer"):
                    prospective.audit(
                        SimpleNamespace(
                            root=root,
                            split=split,
                            parent_bank=None,
                            source_package=None,
                            previous_plan=None,
                        )
                    )
            prospective.report(SimpleNamespace(root=root, out=root / "report"))
            summary = read(root / "report/summary.json")
            self.assertEqual(summary["pooled"]["episodes"], 2)
            self.assertEqual(summary["pooled"]["complete_pairs"], 1)
            self.assertEqual(
                summary["applicable_by_action_exposure"]["exposed"]["four_cells"]["rescue"], 1
            )
            self.assertEqual(summary["applicable_by_action_exposure"]["avoided"]["episodes"], 0)
            self.assertEqual(
                summary["strata"]["target_train"]["free_aa_agree_given_applicable"]["rate"], 1
            )
            pair = read(root / "audit-target_train/pairs/pairs.json")[0]
            self.assertEqual(
                (pair["arms"]["silent"]["revision_calls"], pair["arms"]["remind"]["revision_calls"]),
                (0, 1),
            )
