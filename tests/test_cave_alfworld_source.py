import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memory_agent.cave import alfworld as aw
from memory_agent.cave import alfworld_memory as memory
from memory_agent.cave.alfworld_env import visible_context
from memory_agent.cave.records import read


def draft(action, tokens=7):
    return {"action": action, "input_tokens": tokens, "output_tokens": 1,
            "prompt_digest": action, "allowed": ["open", "close"], "seconds": 0.}


class Episode:
    def __init__(self, *unused):
        self.history = []
        self.initial_signature = "s0"
        self.env = SimpleNamespace(close=lambda: None)
        self.goal, self.observation, self.commands = "put a book in bin", "", ["open", "close"]

    def signature(self):
        return "s" + str(len(self.history))

    def restore(self, prefix, signature):
        other = Episode()
        for r in prefix:
            other.advance(r["action"])
        if other.signature() != signature:
            raise ValueError("invalid test checkpoint")
        return other

    def advance(self, action):
        row = {"action": action, "observation": "", "seconds": 0.,
               "state": "s" + str(len(self.history) + 1)}
        self.history.append(row)
        return row

    @property
    def terminal(self):
        return len(self.history) == 2

    @property
    def won(self):
        return self.history[0]["action"] == "open"


class Actor:
    info = {"actor": "fixture"}

    def __init__(self):
        self.calls = []

    def propose(self, ep, reminder=None, original=None, *, seed=None, temperature=None):
        self.calls.append((len(ep.history), seed, temperature))
        return draft("open" if reminder else "close")


class SourceReplayTests(unittest.TestCase):
    def test_registration_freezes_four_groups_and_disjoint_diagnostic(self):
        with tempfile.TemporaryDirectory() as root, patch.object(aw, "FAMILIES", ("example",)):
            root = Path(root)
            (root / "fake.safetensors").write_bytes(b"weights")
            config = root / "config.yaml"
            config.write_text("cfg")
            for split in ("train", "valid_seen", "valid_unseen"):
                for i in range(9):
                    path = root / "json_2.1.1" / split / f"example-{split}{i}-2/trial/game.tw-pddl"
                    aw.save(path, {"solvable": True, "fixture": str(path)})
            args = SimpleNamespace(data=root, weights=root, out=root/"out", config=config,
                                   source_replicates=3, source_temperature=.7,
                                   single_arm=True, inspect_per_family=1)
            aw.register(args)
            manifest = read(args.out / "manifest.json")
            self.assertEqual(manifest["policy"]["train_collection"], "single_arm")
            self.assertEqual(manifest["policy"]["controller_version"], 2)
            self.assertEqual(manifest["policy"]["source_replay"]["k"], 3)
            groups = manifest["splits"]
            self.assertEqual({s: len(v) for s, v in groups.items()},
                             {"train": 2, "source": 3, "test": 2, "validation": 1, "diagnostic": 1})
            self.assertEqual(9, len({r["template"] for group in groups.values() for r in group}))
            self.assertEqual(manifest["model_files"], aw.weight_fingerprint(root))
            (root / "fake.safetensors").write_bytes(b"changed")
            self.assertNotEqual(manifest["model_files"], aw.weight_fingerprint(root))

    def test_k_seed_coupling_economics_and_bank(self):
        actor = Actor()
        natural = aw.roll(Episode(), actor)
        card = {"step": 1, "action": "open", "reminder": "Try opening",
                "phase": "search", "scope": "any", "unvisited": False,
                "target": "book", "family": "test", "pattern": "open", "mechanism": "blocked"}
        protocol = {"version": 2, "confidence": .95, "k": 3, "temperature": .7}
        spec = {"id": "source-01", "seed": 99, "sha256": "game"}
        manifest = {"config_sha256": "cfg", "splits": {"source": [spec]},
                    "policy": {"source_replay": protocol}}
        with tempfile.TemporaryDirectory() as root, patch.object(memory, "Episode", Episode):
            folder = Path(root) / "source/source-01"
            aw.save(folder / "natural.json", natural)
            aw.save(folder / "proposal.json", {"status": "ok", "natural_digest": aw.digest(natural),
                                               "candidate": {"accepted": [card]}})
            args = SimpleNamespace(out=Path(root), data=None, config=None)
            bank = memory.build_bank(args, manifest, actor)
            evidence = read(folder / "replay-1.json")
            self.assertEqual(bank[0]["q"]["direct"]["ci95"], [1, 1])
            self.assertEqual(bank[0]["original_action"], "close")
            self.assertEqual(3, len(set(r["seed"] for r in evidence["replicates"])))
            suffix_seeds = [seed for step, seed, temp in actor.calls if step == 1 and temp == .7]
            self.assertEqual(9, len(suffix_seeds))
            self.assertEqual([3, 3, 3], sorted(suffix_seeds.count(s) for s in set(suffix_seeds)))
            for r in evidence["replicates"]:
                s, d, m = (r["metrics"][a] for a in ("silent", "direct", "remind"))
                self.assertEqual(s["tokens"], d["tokens"])
                self.assertEqual(s["actor_calls"], d["actor_calls"])
                self.assertEqual(m["tokens"], s["tokens"] + 8)
                self.assertEqual(m["actor_calls"], s["actor_calls"] + 1)
            self.assertEqual(memory.build_bank(args, manifest, actor), bank)
            protocol["k"] = 2
            with self.assertRaisesRegex(ValueError, "provenance"):
                memory.build_bank(args, manifest, actor)

    def test_single_seed_uncertainty_and_duplicate_rejection(self):
        record = {"seed": 42, "metrics": {"silent": {"utility": 0.}, "direct": {"utility": 1.}}}
        out = memory.paired_contrast([record], "direct")
        self.assertIsNone(out["ci95"])
        self.assertFalse(out["supports_gain"])
        with self.assertRaisesRegex(ValueError, "distinct"):
            memory.paired_contrast([record, record], "direct")

    def test_legacy_bank_can_be_read_without_running_v2(self):
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(out=Path(root))
            aw.save(args.out / "bank.json", ["legacy"])
            self.assertEqual(memory.build_bank(args, {"policy": {}}, Actor()), ["legacy"])

    def test_teacher_gets_only_successful_source_comparison(self):
        actor = Actor()
        fail = aw.roll(Episode(), actor)
        success = aw.roll(Episode(), actor, first=draft("open"))
        specs = [{"id": "failed", "family": "easy"}, {"id": "success", "family": "easy"}]
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(out=Path(root))
            for spec, trace in zip(specs, [fail, success]):
                aw.save(args.out / "source" / spec["id"] / "natural.json", trace)
            with patch.object(memory, "teacher_request", return_value={"candidate": {"accepted": []}}) as teacher:
                memory.propose_all(args, {"policy": {"source_replay": {}}, "splits": {"source": specs}})
            payload = teacher.call_args.args[0]
            self.assertEqual([x["id"] for x in payload["successful_source_references"]], ["success"])
            self.assertEqual(len(payload["steps"]), len(fail["steps"]))

    def test_public_holding_state_after_move(self):
        ep = Episode()
        ep.goal = "look at book under the desklamp"
        ep.history = [{"action": "take book 1 from shelf 1", "observation": "You pick up the book 1."}]
        self.assertEqual(visible_context(ep)["phase"], "holding_target")
        ep.history.append({"action": "move book 1 to shelf 1", "observation": "You move the book 1."})
        self.assertEqual(visible_context(ep)["phase"], "search")
