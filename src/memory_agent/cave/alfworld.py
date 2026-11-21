"""Run an independent ALFWorld cohort through CAVE's source and target stages."""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from .alfworld_env import MAX_STEPS, Actor, Episode, sha_file, visible_context
from .records import digest, read, save


FAMILIES = ("pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
            "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place")
QUOTAS = {"test": ("valid_unseen", 2), "validation": ("valid_seen", 1),
          "train": ("train", 2), "source": ("train", 3)}
SEED = 20261121
PRICES = {"per_1000_tokens": 0.01, "per_step": 0.001, "per_minute": 0.01}


def weight_fingerprint(folder):
    shards = sorted(folder.glob("*.safetensors"))
    if not shards:
        raise ValueError("missing frozen Qwen weight shards")
    files = [*shards, *(p for p in sorted(folder.glob("*.json")) if p.is_file())]
    return {p.name: sha_file(p) for p in files}


def controller_sources():
    folder = Path(__file__).parent
    modules = ("alfworld", "alfworld_controller", "alfworld_env", "alfworld_experiment",
               "alfworld_memory", "auxiliary_heads", "constrained_gate", "gate_mlp",
               "intervention_policy", "logged_policy_fit", "models", "probe_value", "records")
    return {name: sha_file(folder / (name + ".py")) for name in modules}


def register(args):
    if args.source_replicates < 1 or not 0 < args.source_temperature <= 2:
        raise ValueError("source replay needs positive K and temperature in (0, 2]")
    if args.inspect_per_family < 0:
        raise ValueError("inspect-per-family must be nonnegative")
    quotas = dict(QUOTAS)
    for stage in ("source", "train", "validation", "test"):
        n = getattr(args, f"{stage}_per_family", None)
        if n is not None:
            if n < 1:
                raise ValueError(f"positive {stage} quota required")
            quotas[stage] = (quotas[stage][0], n)
    if args.inspect_per_family:
        quotas["diagnostic"] = ("train", args.inspect_per_family)
    data = args.data / "json_2.1.1"
    buckets = defaultdict(list)
    for split in {value[0] for value in QUOTAS.values()}:
        for path in sorted((data / split).glob("*/*/game.tw-pddl")):
            template = path.parent.parent.name.rsplit("-", 1)[0]
            family = template.split("-", 1)[0]
            if family in FAMILIES and "Sliced" not in template:
                buckets[split, family].append((path, template))
    rng = random.Random(SEED)
    used, hashes = set(), set()
    splits = {}
    heldout = set(getattr(args, "holdout_family", ()))
    if heldout and (heldout == set(FAMILIES) or not heldout.issubset(FAMILIES)):
        raise ValueError("held-out families must leave at least one development family")
    for stage, (split, quota) in quotas.items():
        selection = []
        for family in FAMILIES:
            if heldout and stage != "source" and ((family in heldout) != (stage == "test")):
                continue
            candidates = list(buckets[split, family])
            rng.shuffle(candidates)
            count = 0
            for path, template in candidates:
                if template in used or not read(path)["solvable"]:
                    continue
                sha = sha_file(path)
                if sha in hashes:
                    continue
                used.add(template)
                hashes.add(sha)
                selection.append({"id": f"{stage}-{len(selection) + 1:02d}",
                                  "game": path.relative_to(args.data).as_posix(), "sha256": sha,
                                  "template": template, "family": family, "seed": rng.randrange(2**32)})
                count += 1
                if count == quota:
                    break
            if count != quota:
                raise ValueError(f"insufficient unique game templates: {stage}/{family}")
        splits[stage] = selection
    policy = {"seed": SEED, "quota_per_family": quotas, "max_steps": MAX_STEPS,
              "heldout_families": sorted(heldout),
              "holdout_mechanism": getattr(args, "holdout_mechanism", False),
              "controller_version": 2, "risk_budget": .05, "extra_token_budget": 256,
              "target_hazard": .5, "training_aligned_hazard": .75, "actor_call_budget": MAX_STEPS + 1,
              "train_collection": "single_arm" if args.single_arm else "paired",
              "source_replay": {"version": 2, "k": args.source_replicates,
                                "temperature": args.source_temperature, "confidence": .95,
                                "interval": "paired Student-t over independent continuation seeds"},
              "utility_prices": PRICES, "target_sampling": "first accepted eligible checkpoint",
              "actor": "Qwen2.5-7B", "teacher": "gpt-6-sol",
              "source_admission": "failed source and positive lower utility contrasts for correction and reminder"}
    save(args.out / "manifest.json", {"policy": policy, "splits": splits,
                                    "config_sha256": sha_file(args.config),
                                    "model_files": weight_fingerprint(args.weights),
                                    "controller_sources": controller_sources()})
    print("REGISTERED", {k: len(v) for k, v in splits.items()}, flush=True)


def snapshot(ep, draft):
    return {"step": len(ep.history) + 1, "state": ep.signature(),
            "context": visible_context(ep), "draft": draft}


def roll(ep, actor, prefix=(), first=None, *, replica_seed=None, temperature=None):
    rows = list(prefix)
    while not ep.terminal:
        draft = first
        if draft is None:
            if replica_seed is None:
                draft = actor.propose(ep)
            else:
                seed = int(digest([replica_seed, "continuation", len(ep.history)])[:15], 16)
                draft = actor.propose(ep, seed=seed, temperature=temperature)
        first = None
        checkpoint = snapshot(ep, draft)
        checkpoint["execution"] = ep.advance(draft["action"])
        rows.append(checkpoint)
    return {"success": int(ep.won), "initial_signature": ep.initial_signature, "steps": rows}


def source_natural(args, manifest, actor, limit):
    for spec in manifest["splits"]["source"][:limit]:
        path = args.out / "source" / spec["id"] / "natural.json"
        if path.exists():
            continue
        ep = Episode(args.data, args.config, spec)
        try:
            natural = roll(ep, actor)
            save(path, natural)
            print("SOURCE_NATURAL", spec["id"], natural["success"], len(natural["steps"]), flush=True)
        finally:
            ep.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pipeline", "register", "natural", "propose", "experiment", "deploy"])
    parser.add_argument("--source-replicates", type=int, default=3, help="matched source continuations per arm")
    parser.add_argument("--source-temperature", type=float, default=.7)
    parser.add_argument("--inspect-per-family", type=int, default=0)
    parser.add_argument("--holdout-family", choices=FAMILIES, action="append", default=[])
    parser.add_argument("--holdout-mechanism", action="store_true",
                        help="reserve one verified failure-signature group for target test only")
    for stage in ("source", "train", "validation", "test"):
        parser.add_argument(f"--{stage}-per-family", type=int)
    parser.add_argument("--single-arm", action="store_true", help="register randomized single-arm training data")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "pipeline":
        from filelock import FileLock

        args.out.mkdir(parents=True, exist_ok=True)
        with FileLock(str(args.out / ".pipeline.lock"), timeout=0):
            return execute(args)
    return execute(args)


def execute(args):
    manifest_path = args.out / "manifest.json"
    if args.stage == "register":
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        register(args)
        return
    if args.stage == "pipeline" and not manifest_path.exists():
        register(args)
    manifest = read(manifest_path)
    if args.stage == "pipeline" and manifest["policy"].get("controller_version") != 2:
        raise ValueError("use a new output directory for the full controller pipeline")
    if manifest["config_sha256"] != sha_file(args.config):
        raise ValueError("environment configuration changed")
    if "controller_sources" in manifest and manifest["controller_sources"] != controller_sources():
        raise ValueError("controller implementation changed since registration")
    if args.stage == "propose":
        from .alfworld_memory import propose_all

        propose_all(args, manifest)
        return
    if "model_files" in manifest and weight_fingerprint(args.weights) != manifest["model_files"]:
        raise ValueError("actor weights or tokenizer differ from registration")
    actor = Actor(args.weights)
    if "model_files" in manifest:
        actor.info["files_digest"] = digest(manifest["model_files"])
    actor_path = args.out / "actor.json"
    if actor_path.exists() and digest(read(actor_path)) != digest(actor.info):
        raise ValueError("actor configuration changed")
    save(actor_path, actor.info)
    if args.stage in ("pipeline", "natural"):
        source_natural(args, manifest, actor, None)
        if args.stage == "natural":
            return
        from .alfworld_memory import propose_all

        propose_all(args, manifest)
    if args.stage == "deploy":
        from .alfworld_experiment import deploy

        deploy(args, manifest, actor)
    else:
        from .alfworld_experiment import run

        run(args, manifest, actor)


if __name__ == "__main__":
    main()
