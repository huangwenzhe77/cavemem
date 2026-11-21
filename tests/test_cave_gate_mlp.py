import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from memory_agent.cave import gate_mlp as m
from memory_agent.cave.records import digest


class TokenizerFixture:
    @classmethod
    def from_file(cls, _path):
        return cls()

    def no_truncation(self):
        pass

    def no_padding(self):
        pass

    def get_vocab_size(self):
        return 65

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return SimpleNamespace(ids=[ord(c) % 64 + 1 for c in text])


def fixture_x(op):
    return {
        "instruction": "Order every number from high to low.",
        "history_before_proposal": [
            {"role": "user", "content": "Current list: [2, -1, 2]"}
        ],
        "working_state": {
            "input": [2, -1, 2],
            "working": [2, -1, 2],
            "submitted": None,
        },
        "memory": {"id": "desc", "reminder": "Keep all values."},
        "proposal": {"commands": [{"keystrokes": c} for c in op], "complete": False},
        "evidence": {"source_verified": 1, "applicable": 1, "exposed": None},
        "budget_after_proposal": {"remaining_actor_calls": 2},
    }


class MLPTests(unittest.TestCase):
    def setUp(self):
        self.tok = TokenizerFixture()
        self.cards = {"desc": "descending"}
        self.table = (
            np.random.default_rng(99)
            .normal(size=(65, m.CONFIG["token_width"]))
            .astype("float32")
        )

    def test_frozen_projection_and_tail_retention(self):
        raw = np.arange(3 * 19, dtype=np.float32).reshape(3, 19)
        order, signs = m.projection(19)
        out = m.project(raw, order, signs)
        expected = np.zeros((3, 16))
        for n, idx in enumerate(order):
            expected[:, n % 16] += raw[:, idx] * signs[n]
        np.testing.assert_allclose(out, expected)
        np.testing.assert_allclose(
            m.pool("abOOPSyz", 4, self.tok, self.table),
            m.pool("abyz", 4, self.tok, self.table),
        )
        self.assertAlmostEqual(
            np.linalg.norm(m.pool("abc", 10, self.tok, self.table)), 1.0
        )

    def test_no_action_channel_and_no_label_contamination(self):
        x = fixture_x(["DESC", "SUBMIT"])
        full = m.encode(x, self.cards, "full", self.tok, self.table)
        without = m.encode(x, self.cards, "no_a0", self.tok, self.table)
        self.assertEqual(len(full), len(without))
        self.assertGreater(np.linalg.norm(full[16:32]), 0)
        np.testing.assert_array_equal(without[16:32], np.zeros(16))
        changed = deepcopy(x)
        del changed["proposal"], changed["evidence"]
        changed.update(parse_warning="new", targets={"D": 999})
        np.testing.assert_array_equal(
            without, m.encode(changed, self.cards, "no_a0", self.tok, self.table)
        )
        x.update(targets={"H": 1}, future_verdict="fail")
        np.testing.assert_array_equal(
            full, m.encode(x, self.cards, "full", self.tok, self.table)
        )

    def test_weighted_fit_and_signed_roundtrip(self):
        feats = [fixture_x(["SHOW"]), fixture_x(["DESC", "SUBMIT"])]
        rows = [
            {
                "sample": {
                    "sample_id": str(i),
                    "targets": {"D": y, "H": 0},
                    "features": feats[k],
                },
                "task_id": f"t{k}",
                "source": "fixture",
                "run_id": "r",
                "checkpoint": str(i),
            }
            for i, (k, y) in enumerate([(0, 0.9), (0, 0.9), (1, -0.02)])
        ]
        plan = {
            "label_spec": {"m_label": 0.002},
            "actor": {},
            "bank_id": "bank",
            "protocol": "engine",
        }

        def fixture_encoder(_qwen, path, _actor):
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")
            np.save(path / "token-embeddings.npy", self.table, allow_pickle=False)
            return {
                "shape": [65, 16],
                "original_tensor_sha256": "fixture",
                "tokenizer_sha256": m.file_sha(path / "tokenizer.json"),
                "table_sha256": m.file_sha(path / "token-embeddings.npy"),
            }

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                m.gate, "load_training", return_value=(rows, self.cards, [], plan)
            ),
            patch.object(m, "export_encoder", side_effect=fixture_encoder),
            patch.dict(
                "sys.modules",
                {
                    "tokenizers": SimpleNamespace(
                        Tokenizer=TokenizerFixture, __version__="fixture"
                    )
                },
            ),
        ):
            out = Path(tmp) / "model"
            args = SimpleNamespace(
                qwen=None,
                out=out,
                formal_root=None,
                supplement_root=None,
                parent_bank=None,
            )
            m.fit(args)
            rt = m.load_model(out / "model.json")
            ledger = m.read(out / "training-ledger.json")
            self.assertEqual([r["weight"] for r in ledger], [0.5, 0.5, 1])
            self.assertEqual(rt.model["train_ledger_digest"], digest(ledger))
            self.assertIsNone(rt.model["risk_head"])
            self.assertGreater(
                rt.predict(feats[0])["predicted_D"], rt.predict(feats[1])["predicted_D"]
            )
            self.assertEqual(
                rt.predict(feats[0], "no_a0"), rt.predict(feats[1], "no_a0")
            )
            self.assertEqual(
                rt.predict(feats[0]), m.load_model(out / "model.json").predict(feats[0])
            )
            with self.assertRaises(ValueError):
                m.fit(args)
            with (out / "token-embeddings.npy").open("ab") as f:
                f.write(b"corruption")
            with self.assertRaises(ValueError):
                m.load_model(out / "model.json")


if __name__ == "__main__":
    unittest.main()
