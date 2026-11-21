"""Train and run an action-conditioned, stochastic, primal-dual reminder policy."""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from . import formal_plan as fp, gate, gate_mlp
from .records import digest, read, save
from .replay import require


CONFIG = {"steps": 400, "lr": .02, "dual_lr": .1, "kl": .04, "temperature": .25,
          "bootstrap": 128, "upper_quantile": .95, "seed": 4261}


def sigmoid(a):
    return np.exp(-np.logaddexp(0., -a))


def upper(weights, v, q):
    """Order statistic and its subgradient for fixed task-cluster bootstrap draws."""
    means = weights @ v
    index = np.argsort(means, kind="stable")[int(np.ceil(q * len(means))) - 1]
    return float(means[index]), weights[index]


def optimize(X, gain, harm, extra, initial, cutoff, limits, eta, config=CONFIG):
    n = len(X)
    require(n >= 2 and X.shape[0] == len(gain) == len(harm) == len(extra)
            and set(harm.tolist()) <= {0, 1} and all(np.isfinite(a).all() for a in (X, gain, extra)),
            "invalid paired task arrays")
    budget = limits["policy_mean_extra_tokens_budget"]
    delta = limits["policy_risk_delta"]
    require(budget > 0 and 0 < delta < 1, "invalid risk/cost limits")
    scale = extra / budget
    rng = np.random.default_rng(config["seed"])
    weights = rng.multinomial(n, np.full(n, 1/n), size=config["bootstrap"]) / n
    params = [p.copy() for p in initial]
    # Reparameterize the final layer so p0=.5 at the supervised gain threshold.
    params[-2] /= config["temperature"]
    params[-1] = (params[-1] - cutoff) / config["temperature"]
    _, initial_logit = gate_mlp.forward(X, params)
    alpha = beta = 0.
    trace = []
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    for step in range(config["steps"] + 1):
        hidden, logit = gate_mlp.forward(X, params)
        prob = sigmoid(logit)
        kl = prob * (logit-initial_logit) + np.logaddexp(0, initial_logit) - np.logaddexp(0, logit)
        risk, rw = upper(weights, prob * harm, config["upper_quantile"])
        cost, cw = upper(weights, prob * scale, config["upper_quantile"])
        if not np.any(harm):
            # The empirical bootstrap is degenerate. Bound P(H=1) on this training distribution.
            risk = max(risk, 1 - (1-config["upper_quantile"])**(1/n))
        if step % 25 == 0 or step == config["steps"]:
            trace.append({"step": step, "gain": float(prob @ gain / n), "KL": float(kl.mean()),
                          "risk_upper_training": risk, "extra_tokens_upper_training": cost*budget,
                          "alpha": alpha, "beta": beta, "mean_remind_probability": float(prob.mean())})
        if step == config["steps"]:
            break
        dz = prob * (1-prob) * ((gain - eta*(logit-initial_logit))/n
                                 - alpha*rw*harm - beta*cw*scale)
        dh = dz[:, None] * params[2] * (1-hidden*hidden)
        gradients = [X.T @ dh, dh.sum(axis=0), hidden.T @ dz, np.array([dz.sum()])]
        for i, grad in enumerate(gradients):
            m[i] = .9*m[i] + .1*grad
            v[i] = .999*v[i] + .001*grad**2
            params[i] += config["lr"]*(m[i]/(1-.9**(step+1))) / (np.sqrt(v[i]/(1-.999**(step+1)))+1e-8)
        alpha = max(0., alpha + config["dual_lr"]*(risk-delta))
        beta = max(0., beta + config["dual_lr"]*(cost-1))
    require(all(np.isfinite(p).all() for p in params), "nonfinite policy")
    return {"parameters": [p.tolist() for p in params], "eta": eta, "trace": trace,
            "train_constraints_met": risk <= delta and cost <= 1}


def vector(base, x):
    return gate_mlp.encode(x, base.model["cards"], "full", base.tokenizer, base.table)


class Policy:
    def __init__(self, path, base_path):
        self.doc = gate.signed(path, "policy_id")
        self.base = gate_mlp.load_model(base_path)
        require(self.doc["base_model_id"] == self.base.model["model_id"]
                and self.doc["protocol"] == fp.code_digest("constrained_gate.py"), "policy provenance")
        self.params = {k: [np.array(a) for a in v["parameters"]] for k, v in self.doc["variants"].items()}
        for params in self.params.values():
            require([p.shape for p in params] == [p.shape for p in self.base.params["full"]]
                    and all(np.isfinite(p).all() for p in params), "policy network shape")

    def probability(self, x, variant="kl"):
        if x is None:
            return 0.
        e = x["evidence"]
        if e["source_verified"] != 1 or e["applicable"] != 1 or x["budget_after_proposal"]["remaining_actor_calls"] < 1:
            return 0.
        _, logit = gate_mlp.forward(vector(self.base, x), self.params[variant])
        return float(sigmoid(logit))

    def decide(self, x, uniform, variant="kl"):
        require(0 <= uniform < 1, "policy decision needs a uniform draw in [0,1)")
        p = self.probability(x, variant)
        return {"p_remind": p, "uniform": uniform, "remind": uniform < p, "policy_id": self.doc["policy_id"]}


def train(outputs, out):
    require(not out.exists(), "new training release required")
    base = gate_mlp.load_model(outputs / "cave-mlp-01/frozen/model.json")
    formal, supplement = outputs / "cave-formal-01", outputs / "cave-supplement-01"
    rows, cards, sources, plan = gate.load_training(formal, supplement, fp.DEFAULT_BANK)
    require(cards == base.model["cards"] and sources == base.model["sources"] and
            len({r["task_id"] for r in rows}) == len(rows), "need one independent paired task per cluster")
    costs = {}
    for source, path in (("formal", formal / "dataset-v1/train-evidence.jsonl"),
                         ("supplement", supplement / "deduplicated-training/evidence.jsonl")):
        for e in gate.lines(path):
            arms = {a: e["arms"][a]["observed"] if source == "formal" else e[a] for a in ("silent", "remind")}
            costs[e["sample_id"]] = arms["remind"]["total_tokens"] - arms["silent"]["total_tokens"]
    samples = [r["sample"] for r in rows]
    X = np.stack([vector(base, s["features"]) for s in samples])
    gain = np.array([s["targets"]["D"] for s in samples])
    harm = np.asarray([s["targets"]["H"] for s in samples], dtype=int)
    extra = np.array([costs[s["sample_id"]] for s in samples])
    variants = {k: optimize(X, gain, harm, extra, base.params["full"], base.model["threshold"],
                             plan["label_spec"], eta) for k, eta in (("kl", CONFIG["kl"]), ("no_kl", 0.))}
    doc = {"schema": "paired-primal-dual/v1", "protocol": fp.code_digest("constrained_gate.py"),
           "base_model_id": base.model["model_id"], "bank_id": plan["bank_id"], "sources": sources,
           "limits": plan["label_spec"], "config": CONFIG, "variants": variants, "task_count": len(rows),
           "train_ledger_digest": base.model["train_ledger_digest"], "risk_events": int(harm.sum()),
           "risk_calibration": "unavailable: zero paired harms" if not harm.any() else "empirical only",
           "scope": "paired training tasks; stochastic gate at an applicable, funded checkpoint"}
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "release"
        save(stage / "model.json", {**doc, "policy_id": digest(doc)})
        loaded = Policy(stage / "model.json", outputs / "cave-mlp-01/frozen/model.json")
        require(0 <= loaded.probability(samples[0]["features"]) <= 1, "release inference failed")
        stage.rename(out)
    print(out / "model.json")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--outputs", type=Path, default=fp.OUTPUT)
    f.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("decide")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--base", type=Path, default=fp.OUTPUT / "cave-mlp-01/frozen/model.json")
    p.add_argument("--uniform", required=True, type=float)
    args = cli.parse_args()
    if args.cmd == "fit":
        train(args.outputs, args.out)
    else:
        print(Policy(args.model, args.base).decide(read(args.features), args.uniform))
