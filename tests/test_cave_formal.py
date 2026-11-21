import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memory_agent.cave import formal, formal_plan as fp, prospective
from memory_agent.cave.records import digest, read, save
from memory_agent.cave.replay import protocol_digest
import test_cave_prospective as fixtures


class FormalTests(unittest.TestCase):
    def test_partition_aliases_and_one_replacement_rule(self):
        cohorts = fp.templates_and_tasks([])
        self.assertEqual(
            [len(cohorts[s]) for s in ("train", "validation", "test")], [48, 24, 16]
        )
        seen = set()
        for rows in cohorts.values():
            groups = {r["task"]["template_id"] for r in rows}
            self.assertFalse(groups & seen)
            seen.update(groups)
        self.assertTrue(fp.near_duplicate([2, 1, 2], [2, 1, -1]))
        self.assertFalse(fp.near_duplicate([2, 1, 2], [2, -1, -1]))
        with self.assertRaises(ValueError):
            fp.templates_and_tasks([cohorts["test"][0]["task"]["input"]])
        with self.assertRaises(ValueError):
            fp.tasks_for({}, "test")
        task = next(
            r["task"]
            for r in cohorts["validation"]
            if r["task"]["family"] == "descending"
        )
        pre = {
            "state": {
                "submitted": None,
                "input": task["input"],
                "working": list(task["input"]),
            }
        }
        draft = {"a0": {"commands": [{"keystrokes": "DESC"}, {"keystrokes": "SUBMIT"}]}}
        self.assertEqual(
            fp.formal_route(fixtures.ProspectiveTests.cards, task, pre, draft)["rho"], 1
        )
        pre["state"]["working"].pop()
        self.assertEqual(
            fp.formal_route(fixtures.ProspectiveTests.cards, task, pre, draft)["rho"], 0
        )
        with self.assertRaises(ValueError):
            fp.formal_route(
                fixtures.ProspectiveTests.cards,
                {**task, "instruction": "Discard negatives"},
                pre,
                draft,
            )

    def test_double_success_measured_cost_and_missing_label(self):
        p = {
            "lambda_tokens": 0.01,
            "lambda_tool": 0.001,
            "lambda_time": 0.01,
            "label_spec": fp.LABELS,
        }
        base = {
            "success": 1,
            "total_tokens": 120,
            "tool_calls": 1,
            "accounted_seconds": 10,
        }
        obs = {
            "arms": {"silent": base, "remind": {**base, "total_tokens": 220}},
            "delta": {"utility": -0.001},
        }
        target = formal.label(obs, p)
        self.assertEqual(
            (target["D"], target["H"], target["hard_label"]), (-0.001, 0, "uncertain")
        )
        self.assertEqual(target["D_cost_sensitivity"]["0"], 0)
        obs["arms"]["remind"]["success"] = None
        with self.assertRaises(ValueError):
            formal.label(obs, p)

    def test_blind_barrier_pair_export_and_seal(self):
        actor = fixtures.ActorFixture()
        actor.info = {**actor.info, "actor": "fixture"}
        groups = fp.templates_and_tasks([])
        cohorts = []
        for s in fp.RUN_SPLITS:
            task = next(
                t
                for t in groups[s]
                if t["task"]["family"] == "descending" and t["main_opportunity"] == 1
            )
            cohorts.append({"split": s, "tasks": [{**task, "episode_id": "ep-0000"}]})
        plan = {
            "actor": actor.info,
            "seed": 129,
            "max_steps": 4,
            "bank_id": "fixture-bank",
            "parent_bank_id": "fixture-up",
            "descending_memory_id": "fixture-down",
            "cohorts": cohorts,
            "router_version": fp.ROUTER,
            "protocol": protocol_digest(),
            "replicates": 1,
            "total_tokens": 16384,
            "total_tools": 64,
            "lambda_tokens": 0.01,
            "lambda_time": 0.01,
            "lambda_tool": 0.001,
            "label_spec": fp.LABELS,
            "label_spec_digest": digest(fp.LABELS),
            "test_manifest_digest": digest({"tasks": []}),
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(fp, "load", return_value=(plan, digest(plan))),
            patch.object(fp, "FrozenQwen", return_value=actor),
            patch.object(prospective, "load_actor", return_value=actor),
            patch.object(
                prospective,
                "verified_cards",
                return_value=fixtures.ProspectiveTests.cards,
            ),
        ):
            root = Path(tmp)
            args = SimpleNamespace(
                root=root,
                split="train",
                parent_bank=None,
                source_package=None,
                previous_plan=None,
                out=root / "dataset",
            )
            save(root / "sealed-test.json", {"tasks": []})
            fp.collect(root, "train")
            with self.assertRaises(FileNotFoundError):
                formal.audit(args)
            fp.collect(root, "validation")
            for s in fp.RUN_SPLITS:
                args.split = s
                formal.audit(args)
            formal.export(args)
            for s in fp.RUN_SPLITS:
                row = json.loads((args.out / (s + ".jsonl")).read_text())
                self.assertIn("proposal", row["features"])
                self.assertNotIn("targets", row["features"])
                self.assertNotIn("score", row["features"])
                self.assertEqual(row["targets"]["hard_loss_weight"], 0)
                self.assertEqual(
                    json.loads((args.out / (s + "-evidence.jsonl")).read_text())[
                        "sample_id"
                    ],
                    row["sample_id"],
                )
            self.assertFalse((args.out / "test.jsonl").exists())
            evidence = root / "audit-train/pairs/pairs.json"
            changed = read(evidence)
            changed[0]["difference"] += 1
            save(evidence, changed)
            args.out = root / "attempt"
            with self.assertRaises(ValueError):
                formal.export(args)
            self.assertFalse(args.out.exists())


if __name__ == "__main__":
    unittest.main()
