import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from memory_agent.cave import alfworld as workflow, alfworld_controller as collector, intervention_policy as policy
from memory_agent.cave.records import digest, read, save
from types import SimpleNamespace


class ToyEpisode:
    def __init__(self, *_):
        self.history = []
        self.env = self
        self.initial_signature = "initial"
        self.observation = self.goal = "put a book in drawer."
        self.commands = ["go to cabinet 1", "go to shelf 1"]
        self.won = self.terminal = False

    def signature(self):
        return str(len(self.history))

    def advance(self, action):
        self.terminal = True
        self.won = action == "go to shelf 1"
        row = {"action": action, "observation": "ended", "state": "end", "seconds": .01}
        self.history.append(row)
        return row

    def close(self):
        pass


class ToyActor:
    def propose(self, ep, reminder=None, original=None):
        return {"action": ep.commands[int(reminder is not None)], "allowed": ep.commands,
                "input_tokens": 6, "output_tokens": 1, "seconds": .2, "prompt_digest": "draft"}


class PolicyFlowTests(unittest.TestCase):
    def test_pipeline_reaches_live_actor_from_fresh_source_failure(self):
        from test_cave_alfworld_source import Actor, Episode
        from memory_agent.cave import alfworld_memory as memory

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            weights = root / "weights"
            weights.mkdir()
            (weights/"fixture.safetensors").write_bytes(b"fixture")
            config = root/"config"
            config.write_text("test")
            for split in ("train", "valid_seen", "valid_unseen"):
                for i in range(9):
                    game = root/"json_2.1.1"/split/f"family-{split}{i}-1"/"trial"/"game.tw-pddl"
                    save(game, {"solvable": True, "fixture": str(game)})
            args = SimpleNamespace(out=root/"out", data=root, weights=weights, config=config,
                                   stage="pipeline", source_replicates=3, source_temperature=.7,
                                   inspect_per_family=1, single_arm=False)
            actor = Actor()
            actor.info = {"actor": "fixture"}

            def teacher(source, endpoint, model, api_style, *, instruction, validate):
                candidate = {"step": 1, "action": "open", "phase": "search", "scope": "any",
                             "unvisited": False, "mechanism": "closed", "reminder": "Open it."}
                return {"candidate": validate(source, {"candidates": [candidate]}),
                        "model": model, "response_model": model, "source_digest": digest(source)}

            args.test_per_family = 1
            with patch.object(workflow, "FAMILIES", ("family",)), patch.object(workflow, "Episode", Episode), \
                 patch.object(workflow, "Actor", return_value=actor), patch.object(memory, "Episode", Episode), \
                 patch.object(memory, "teacher_request", side_effect=teacher), patch.object(collector, "Episode", Episode):
                workflow.execute(args)
                report = read(args.out/"summary.json")
                self.assertEqual(report["source"]["admitted"], 3)
                self.assertEqual(report["source"]["paired_replicates"], 9)
                self.assertEqual(report["live"]["games"], 1)
                self.assertTrue((args.out/"controller"/"frozen.json").is_file())
                replay_count = len(actor.calls)
                workflow.execute(args)
                self.assertEqual(len(actor.calls), replay_count)

    def test_deployment_controls_the_only_real_action(self):
        card = {"id": "src", "source_original": "go to cabinet 2", "source_menu": ["go to shelf 1"],
                "source_checked": [], "scope": "any", "phase": "search", "unvisited": True,
                "pattern": "go to shelf #", "reminder": "Consider the shelf."}
        spec = {"id": "test-01", "family": "simple", "seed": 11}
        config = {"target_hazard": 1.}
        with tempfile.TemporaryDirectory() as temp, patch.object(collector, "Episode", ToyEpisode):
            for remind in (True, False):
                path = Path(temp)/str(remind)/"result.json"
                action = {"remind": remind, "probability": float(remind), "reason": "test"}
                with patch.object(policy, "decide", return_value=action):
                    row = collector.collect_one(NoneArgs(), spec, ToyActor(), [card], config,
                                                "live", path, ({}, None))
                a = "remind" if remind else "silent"
                self.assertEqual(set(row["arms"]), {a})
                self.assertEqual(row["arms"][a]["success"], int(remind))
                self.assertEqual(row["arms"][a]["actor_calls"], 1+int(remind))
                self.assertEqual(read(path.parent/"blind"/"001.json")["draft"]["action"], "go to cabinet 1")
                self.assertEqual(read(path.parent/"before-execution.json")["assignment"]["arm"], a)
            logged = collector.collect_one(NoneArgs(), spec, ToyActor(), [card], config,
                                           "single_arm", Path(temp)/"logged"/"result.json")
            self.assertEqual(logged["assignment"]["propensity"], .5)
            self.assertNotIn("delta", logged)
            self.assertEqual(len(logged["arms"]), 1)

    def test_training_both_estimators_and_heldout_gate(self):
        rows = [{"spec": {"template": str(i), "id": str(i)}, "match": {"e": i%2},
                 "arms": {"silent": {"success": 0, "tokens": 12, "utility": .01},
                          "remind": {"success": 1, "tokens": 18, "utility": 1.}},
                 "delta": {"utility": .99, "tokens": 6}} for i in range(240)]
        vec = lambda r: np.array([r["match"]["e"], 1.])
        limits = {"risk_budget": .05, "extra_token_budget": 256}
        for kind in ("paired", "logged_DR"):
            if kind == "logged_DR":
                for i,r in enumerate(rows):
                    arm = ("remind", "silent")[i%2]
                    r["assignment"] = {"arm": arm, "propensity": .5}
                    r["arms"] = {arm: r["arms"][arm]}
                    del r["delta"]
            model = policy.train(rows[:100], vec, limits)
            self.assertEqual(model["estimator"], kind)
            model["calibration"] = policy.calibrate(model, rows[100:], vec)
            decision = policy.decide(model, vec(rows[0]), .0)
            if kind == "paired":
                self.assertTrue(model["calibration"]["enabled"])
                self.assertTrue(decision["remind"])
            else:
                self.assertEqual(len(model["crossfit"]), 100)
                self.assertEqual(model["risk_estimand"], "possible_harm_upper_bound")
            model["calibration"] = policy.calibrate(model, [], vec)
            self.assertFalse(policy.decide(model, vec(rows[0]), .0)["remind"])


class NoneArgs:
    config = data = None
