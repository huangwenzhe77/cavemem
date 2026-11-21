import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_cave_prospective as fixtures
from memory_agent.cave import formal_plan as fp, prospective, supplement as sup
from memory_agent.cave.records import digest, read, save
from memory_agent.cave.replay import protocol_digest


class SupplementTests(unittest.TestCase):
    prices = {"lambda_tokens": 0.01, "lambda_time": 0.01, "lambda_tool": 0.001}

    def test_strict_intersection_and_cells(self):
        s = {"success": 1, "total_tokens": 100, "accounted_seconds": 5, "tool_calls": 9}
        r = {**s, "total_tokens": 150, "accounted_seconds": 6}

        def group(s, r):
            return sup.classify({"silent": s, "remind": r}, self.prices)[
                "target_subset"
            ]

        self.assertEqual(group(s, r), "costly-both-pass")
        self.assertEqual(group(s, {**r, "success": 0}), "costly-regression")
        for field in ("total_tokens", "accounted_seconds"):
            self.assertIsNone(group(s, {**r, field: s[field]}))
        self.assertIsNone(group(s, {**r, "tool_calls": 7}))
        self.assertIsNone(group({**s, "success": 0}, r))
        self.assertIsNone(group(s, {**r, "success": None}))

    def test_one_train_seal_replay_and_disjoint_views(self):
        actor = fixtures.ActorFixture()
        actor.info = {**actor.info, "actor": "fixture"}
        group = next(
            t
            for t in fp.templates_and_tasks([])["train"]
            if t["task"]["family"] == "descending" and t["main_opportunity"] == 1
        )
        group = {**group, "input_stratum": "ordinary", "episode_id": "ep-0000"}
        plan = {
            "actor": actor.info,
            "seed": 33,
            "max_steps": 4,
            "total_tokens": 16384,
            "total_tools": 64,
            "bank_id": "bank",
            "parent_bank_id": "parent",
            "descending_memory_id": "desc",
            "replicates": 1,
            "cohorts": [{"split": "train", "tasks": [group]}],
            "router_version": fp.ROUTER,
            "protocol": protocol_digest(),
            "label_spec": fp.LABELS,
            "label_spec_digest": digest(fp.LABELS),
            "test_manifest_digest": "test",
            "parent_dataset_id": "original-dataset",
            "parent_plan_id": "original-plan",
            **self.prices,
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
            patch.object(sup, "validate_plan", return_value=(plan, digest(plan))),
        ):
            root = Path(tmp)
            args = SimpleNamespace(
                root=root,
                parent=None,
                parent_bank=None,
                source_package=None,
                previous_plan=None,
                out=root / "release",
            )
            fp.collect(root, "train")
            sup.audit(args)
            sup.export(args)
            summary = read(args.out / "dataset.json")["summary"]
            self.assertEqual(summary["complete_pairs"], 1)
            self.assertEqual(summary["cells"]["rescue"], 1)
            self.assertEqual(summary["other_complete_pairs"], 1)
            ex = sup.jsonlines(args.out / "train.jsonl")[0]
            ev = sup.jsonlines(args.out / "train-evidence.jsonl")[0]
            self.assertEqual(ev["sample_id"], ex["sample_id"])
            self.assertEqual(ex["targets"]["hard_label"], "uncertain")
            self.assertNotIn("cell", ex["features"])
            self.assertTrue(
                all(not sup.jsonlines(args.out / (g + ".jsonl")) for g in sup.TARGETS)
            )
            self.assertFalse((root / "validation").exists())
            raw = root / "audit-train/pairs/pairs.json"
            save(raw, [])
            args.out = root / "corrected"
            with self.assertRaises(ValueError):
                sup.export(args)
            self.assertFalse(args.out.exists())


if __name__ == "__main__":
    unittest.main()
