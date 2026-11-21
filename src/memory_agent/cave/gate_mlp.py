"""Train a small gain head over frozen Qwen input-token embeddings and prefix features."""

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import formal_plan as fp, gate
from .records import digest, read, save
from .replay import require


CONFIG = {
    "seed": 9009,
    "token_width": 16,
    "context_tokens": 192,
    "action_tokens": 64,
    "hidden": 8,
    "steps": 400,
    "learning_rate": 0.01,
    "l2": 0.001,
    "adam_betas": [0.9, 0.999],
    "adam_epsilon": 1e-8,
    "threshold": 0.002,
    "loss": "Task-weighted MSE(D) plus L2/2 times sum of squared non-bias weights",
    "optimizer": "full-batch Adam, fixed updates, no validation or early stopping",
    "text": "Instruction, frozen reminder, latest user message from Hpre. Separate current parsed command batch and completion flag.",
    "encoder": "Qwen input-token matrix, signed balanced column buckets, mean over token rows, per-field unit L2 norm",
}
STRUCT = tuple(gate.BASE) + tuple(gate.ACTION)
INPUT_WIDTH = 2 * CONFIG["token_width"] + len(STRUCT)
EMBEDDING_NAME = "model.embed_tokens.weight"


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def projection(width):
    rng = np.random.default_rng(CONFIG["seed"])
    return rng.permutation(width), rng.choice(
        np.array([-1, 1], dtype=np.float32), width
    )


def project(block, order, signs):
    width = CONFIG["token_width"]
    folded = block[:, order] * signs
    extra = (-folded.shape[1]) % width
    if extra:
        folded = np.pad(folded, ((0, 0), (0, extra)))
    # A common scale has no effect after the per-field norm.
    return folded.reshape(len(block), -1, width).sum(axis=1, dtype=np.float32)


def export_encoder(qwen, dest, actor):
    import torch
    from safetensors import safe_open

    require(
        qwen.resolve() == Path(actor["actor"]).resolve(),
        "expected the frozen actor's local model directory",
    )
    files = [
        {"name": p.name, "bytes": p.stat().st_size}
        for p in sorted(qwen.glob("*.safetensors"))
    ]
    require(
        files == actor["weights"],
        "Qwen shard names or byte sizes differ from the frozen actor",
    )
    cfg = read(qwen / "config.json")
    index = read(qwen / "model.safetensors.index.json")["weight_map"]
    name = index[EMBEDDING_NAME]
    require(
        Path(name).name == name
        and name in {f["name"] for f in files}
        and cfg["model_type"] == actor["model_type"],
        "unexpected embedding source",
    )
    h = hashlib.sha256()
    with safe_open(str(qwen / name), framework="pt", device="cpu") as store:
        matrix = store.get_slice(EMBEDDING_NAME)
        vocab, width = matrix.get_shape()
        require(
            [vocab, width] == [cfg["vocab_size"], cfg["hidden_size"]],
            "embedding and config dimensions differ",
        )
        table = np.empty((vocab, CONFIG["token_width"]), dtype=np.float32)
        order, signs = projection(width)
        for start in range(0, vocab, 512):
            chunk = matrix[start : start + 512].contiguous()
            h.update(chunk.view(torch.uint8).numpy().tobytes())
            table[start : start + len(chunk)] = project(
                chunk.float().numpy(), order, signs
            )
    require(np.isfinite(table).all(), "nonfinite projected token vectors")
    np.save(dest / "token-embeddings.npy", table, allow_pickle=False)
    shutil.copyfile(qwen / "tokenizer.json", dest / "tokenizer.json")
    return {
        "tensor": EMBEDDING_NAME,
        "source_shard": name,
        "shape": [vocab, width],
        "tensor_dtype": str(chunk.dtype),
        "original_tensor_sha256": h.hexdigest(),
        "original_config_sha256": file_sha(qwen / "config.json"),
        "tokenizer_sha256": file_sha(dest / "tokenizer.json"),
        "table_sha256": file_sha(dest / "token-embeddings.npy"),
    }


def load_encoder(path, source):
    from tokenizers import Tokenizer

    require(
        file_sha(path / "tokenizer.json") == source["tokenizer_sha256"]
        and file_sha(path / "token-embeddings.npy") == source["table_sha256"],
        "tokenizer or frozen table changed",
    )
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    table = np.load(path / "token-embeddings.npy", allow_pickle=False)
    require(
        table.shape == (source["shape"][0], CONFIG["token_width"])
        and table.dtype == np.float32
        and tokenizer.get_vocab_size() <= len(table)
        and np.isfinite(table).all(),
        "invalid token table",
    )
    table.flags.writeable = False
    return tokenizer, table


def pool(text, cap, tokenizer, table):
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if len(ids) > cap:
        ids = ids[: cap // 2] + ids[-(cap - cap // 2) :]
    if not ids:
        return np.zeros(CONFIG["token_width"])
    v = table[ids].mean(axis=0, dtype=np.float64)
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def encode(x, cards, variant, tokenizer, table):
    flags = gate.features_for(x, cards, variant)
    previous = next(
        (
            m["content"]
            for m in reversed(x["history_before_proposal"])
            if m["role"] == "user"
        ),
        "",
    )
    prefix = x["instruction"] + "\n" + x["memory"]["reminder"] + "\n" + previous
    ctx = pool(prefix, CONFIG["context_tokens"], tokenizer, table)
    action = np.zeros(CONFIG["token_width"])
    if variant == "full":
        a = x["proposal"]
        text = (
            "\n".join(c["keystrokes"].strip() for c in a["commands"])
            + "\ncomplete="
            + str(bool(a["complete"])).lower()
        )
        action = pool(text, CONFIG["action_tokens"], tokenizer, table)
    return np.concatenate((ctx, action, [float(flags.get(k, False)) for k in STRUCT]))


def initialize(input_width=INPUT_WIDTH):
    rng = np.random.default_rng(CONFIG["seed"])
    h = CONFIG["hidden"]
    return [
        rng.uniform(
            -math.sqrt(6 / (input_width + h)),
            math.sqrt(6 / (input_width + h)),
            (input_width, h),
        ),
        np.zeros(h),
        rng.uniform(-math.sqrt(6 / (h + 1)), math.sqrt(6 / (h + 1)), h),
        np.zeros(1),
    ]


def forward(X, params):
    w, b, out, bias = params
    hidden = np.tanh(X @ w + b)
    return hidden, hidden @ out + bias[0]


def train_head(X, y, weights):
    params = initialize(X.shape[1])
    m = [np.zeros_like(v) for v in params]
    v = [np.zeros_like(p) for p in params]
    weights = weights / weights.sum()
    beta1, beta2 = CONFIG["adam_betas"]
    for step in range(1, CONFIG["steps"] + 1):
        hidden, pred = forward(X, params)
        d = 2 * weights * (pred - y)
        local = d[:, None] * params[2] * (1 - hidden * hidden)
        grads = [
            X.T @ local + CONFIG["l2"] * params[0],
            local.sum(axis=0),
            hidden.T @ d + CONFIG["l2"] * params[2],
            np.array([d.sum()]),
        ]
        for i, g in enumerate(grads):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            params[i] -= (
                CONFIG["learning_rate"]
                * (m[i] / (1 - beta1**step))
                / (np.sqrt(v[i] / (1 - beta2**step)) + CONFIG["adam_epsilon"])
            )
    _, pred = forward(X, params)
    mse = float(weights @ ((pred - y) ** 2))
    require(np.isfinite(mse), "training produced nonfinite gain predictions")
    return {"parameters": [p.tolist() for p in params], "weighted_training_mse": mse}


def fit(args):
    require(not args.out.exists(), "refusing to replace a frozen MLP")
    rows, cards, sources, plan = gate.load_training(
        args.formal_root, args.supplement_root, args.parent_bank
    )
    require(
        rows and plan["label_spec"]["m_label"] == CONFIG["threshold"],
        "empty or incompatible training plan",
    )
    counts = Counter(r["task_id"] for r in rows)
    weights = np.asarray([1 / counts[r["task_id"]] for r in rows])
    y = np.asarray([r["sample"]["targets"]["D"] for r in rows])
    ledger = [
        {
            "sample_id": r["sample"]["sample_id"],
            "task_id": r["task_id"],
            "source": r["source"],
            "run_id": r["run_id"],
            "checkpoint": r["checkpoint"],
            "weight": float(w),
        }
        for r, w in zip(rows, weights, strict=True)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "model"
        stage.mkdir()
        source = export_encoder(args.qwen, stage, plan["actor"])
        tokenizer, table = load_encoder(stage, source)
        variants = {}
        for variant in ("full", "no_a0"):
            X = np.stack(
                [
                    encode(r["sample"]["features"], cards, variant, tokenizer, table)
                    for r in rows
                ]
            )
            variants[variant] = train_head(X, y, weights)
        import tokenizers

        m = {
            "schema": "frozen-token-mlp-gain/v1",
            "fit_code": fp.code_digest("gate_mlp.py"),
            "gate_code": fp.code_digest("gate.py"),
            "engine": plan["protocol"],
            "bank_id": plan["bank_id"],
            "actor_digest": digest(plan["actor"]),
            "tokenizers_version": tokenizers.__version__,
            "numpy_version": np.__version__,
            "encoder": source,
            "cards": cards,
            "config": CONFIG,
            "threshold": CONFIG["threshold"],
            "variants": variants,
            "risk_head": None,
            "risk_events": sum(r["sample"]["targets"]["H"] for r in rows),
            "risk_status": "not_fitted",
            "sources": sources,
            "train_ledger_digest": digest(ledger),
            "sample_count": len(rows),
            "task_ids": sorted(counts),
            "input_width": INPUT_WIDTH,
            "structural_slots": list(STRUCT),
            "label_spec": plan["label_spec"],
        }
        save(stage / "training-ledger.json", ledger)
        save(stage / "model.json", {**m, "model_id": digest(m)})
        stage.rename(args.out)
    print(
        "frozen MLP",
        digest(m),
        "samples",
        len(y),
        "Qwen embedding digest",
        source["original_tensor_sha256"],
    )


class Runtime:
    def __init__(self, model, tokenizer, table):
        self.model = model
        self.tokenizer = tokenizer
        self.table = table
        self.params = {}
        for variant in ("full", "no_a0"):
            params = [
                np.asarray(a, dtype=np.float64)
                for a in model["variants"][variant]["parameters"]
            ]
            require(
                [p.shape for p in params]
                == [
                    (INPUT_WIDTH, CONFIG["hidden"]),
                    (CONFIG["hidden"],),
                    (CONFIG["hidden"],),
                    (1,),
                ]
                and all(np.isfinite(p).all() for p in params),
                "invalid MLP parameter dimensions",
            )
            self.params[variant] = params

    def predict(self, features, variant="full"):
        x = encode(features, self.model["cards"], variant, self.tokenizer, self.table)
        _, gain = forward(x, self.params[variant])
        d = float(gain)
        return {
            "remind": d > self.model["threshold"],
            "predicted_D": d,
            "empirical_H": None,
        }


def load_model(path):
    start = time.perf_counter()
    path = Path(path)
    m = gate.signed(path, "model_id")
    require(
        m["schema"] == "frozen-token-mlp-gain/v1"
        and m["fit_code"] == fp.code_digest("gate_mlp.py")
        and m["gate_code"] == fp.code_digest("gate.py")
        and m["config"] == CONFIG
        and m["train_ledger_digest"]
        == digest(read(path.parent / "training-ledger.json")),
        "model provenance changed",
    )
    tokenizer, table = load_encoder(path.parent, m["encoder"])
    runtime = Runtime(m, tokenizer, table)
    runtime.load_seconds = time.perf_counter() - start
    return runtime


def main():
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--qwen", type=Path, required=True)
    f.add_argument("--formal-root", type=Path, default=fp.OUTPUT / "cave-formal-01")
    f.add_argument(
        "--supplement-root", type=Path, default=fp.OUTPUT / "cave-supplement-01"
    )
    f.add_argument("--parent-bank", type=Path, default=fp.DEFAULT_BANK)
    f.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("predict")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--variant", choices=("full", "no_a0"), default="full")
    args = cli.parse_args()
    if args.command == "fit":
        fit(args)
    else:
        print(
            json.dumps(
                load_model(args.model).predict(read(args.features), args.variant)
            )
        )


if __name__ == "__main__":
    main()
