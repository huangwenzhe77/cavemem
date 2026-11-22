"""Build a source-only demonstration corpus and register official ALFWorld splits."""

import argparse
import contextlib
import io
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .alfworld import FAMILIES
from .alfworld_env import MAX_STEPS, sha_file
from .records import read, save


def available(data, bucket):
    for path in sorted((data/"json_2.1.1"/bucket).glob("*/*/game.tw-pddl")):
        family = path.parent.parent.name.split("-",1)[0]
        if family in FAMILIES and "Sliced" not in path.as_posix() and read(path)["solvable"]:
            yield {"file":path.relative_to(data).as_posix(),"family":family,
                   "template":path.parent.parent.name.rsplit("-",1)[0],"file_sha256":sha_file(path)}


def register(args):
    path = args.out/"manifest.json"
    if path.exists():
        raise FileExistsError(path)
    rng = random.Random(args.seed)
    train = list(available(args.data,"train"))
    rng.shuffle(train)
    splits = {s:[] for s in ("demonstration","historical","development","valid_seen","valid_unseen")}
    used = set()
    for family in FAMILIES:
        pool = [r for r in train if r["family"]==family]
        for name, quantity in (("historical",5),("development",4),("demonstration",args.demo_per_family)):
            rows = []
            for r in pool:
                if r["template"] not in used:
                    rows.append(r)
                    if name!="demonstration":
                        used.add(r["template"])
                    if len(rows)==quantity:
                        break
            if len(rows)!=quantity:
                raise ValueError(f"insufficient {name}/{family}")
            if name=="demonstration":
                used.update(r["template"] for r in rows)
            splits[name].extend(rows)
    for split, expected in (("valid_seen",140),("valid_unseen",134)):
        rows = list(available(args.data,split))
        if len(rows)!=expected:
            raise ValueError(f"official {split}: {len(rows)} games, expected {expected}")
        splits[split] = rows
    for stage, rows in splits.items():
        for i,r in enumerate(rows,1):
            r.update(id=f"{stage}-{i:03d}", seed=rng.randrange(2**32),
                     game=r.pop("file"),sha256=r.pop("file_sha256"))
    save(path,{"splits":splits,"seed":args.seed,"config_sha256":sha_file(args.config),
               "evaluation_unit":"all official TextWorld games of each split",
               "training_only_expert":True})
    print("REGISTERED", {k:len(v) for k,v in splits.items()}, flush=True)


_RECOVERY_RANKER = None


def load_recovery(root):
    global _RECOVERY_RANKER
    from .alfworld_ranker import load

    _RECOVERY_RANKER = load(root)


def demonstrate(payload):
    import yaml
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    data, config, spec, out, recovery = payload
    dest = out/("recovery" if recovery else "demos")/(spec["id"]+".json")
    if dest.exists():
        d = read(dest)
        return d["success"], d["status"]
    game = data/spec["game"]
    game.relative_to(data/"json_2.1.1"/"train")
    if sha_file(game)!=spec["sha256"]:
        raise ValueError("changed training game file")
    os.environ["ALFWORLD_DATA"] = str(data)
    cfg = yaml.safe_load(config.read_text())
    cfg["dataset"]["data_path"] = str(game.parent)
    cfg["env"].update(domain_randomization=False,expert_type="handcoded")
    cfg["general"]["training_method"] = "dagger"
    cfg["dagger"]["training"]["max_nb_steps_per_episode"] = MAX_STEPS
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        system = AlfredTWEnv(cfg)
        if system.game_files != [str(game)]:
            raise ValueError("unexpected expert game selection")
        env = system.init_env(batch_size=1)
        env.seed(spec["seed"])
        rng = random.Random(spec["seed"])
        rows, success, status = [], False, "horizon"
        try:
            obs, info = env.reset()
            goal = obs[0].partition("Your task is to:")[2].strip()
            for _ in range(MAX_STEPS):
                action = info["extra.expert_plan"][0][0]
                commands = info["admissible_commands"][0]
                if action not in commands:
                    status = "expert_invalid"
                    break
                selected = action
                if recovery and rng.random()<.45:
                    from types import SimpleNamespace

                    view = SimpleNamespace(goal=goal,observation=obs[0],commands=commands,
                                           history=[{"action":r["executed"]} for r in rows])
                    ranked = _RECOVERY_RANKER.ordering(view)
                    selected = rng.choice(ranked[:min(3,len(ranked))])[0]
                rows.append({"observation":obs[0],"commands":commands,"action":action,"executed":selected})
                obs, _, done, info = env.step([selected])
                success = bool(info["won"][0])
                if done[0] or success:
                    status = "passed" if success else "terminated"
                    break
        except (RuntimeError, KeyError, IndexError, ValueError) as exc:
            status = type(exc).__name__
            goal = locals().get("goal","")
        finally:
            env.close()
    save(dest,{"spec":spec,"goal":goal,"rows":rows,"success":success,"status":status})
    return success,status


def corpus(args, manifest):
    rows = manifest["splits"]["demonstration"]
    recovery = args.stage=="recovery"
    with ProcessPoolExecutor(max_workers=args.workers,initializer=load_recovery if recovery else None,
                             initargs=(args.recovery_ranker,) if recovery else ()) as executor:
        receipts = executor.map(demonstrate, ((args.data,args.config,s,args.out,recovery) for s in rows))
        successes = 0
        for i,(ok,status) in enumerate(receipts,1):
            successes += ok
            if i%10==0 or i==len(rows):
                print("RECOVERY" if recovery else "EXPERT",i,len(rows),successes,status,flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=["register","expert","ranker","recovery"])
    cli.add_argument("--data",type=Path,required=True)
    cli.add_argument("--config",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--seed",type=int,default=262733)
    cli.add_argument("--demo-per-family",type=int,default=28)
    cli.add_argument("--workers",type=int,default=4)
    cli.add_argument("--rich-ranker",action="store_true")
    cli.add_argument("--lexical-ranker",action="store_true")
    cli.add_argument("--recovery-ranker",type=Path)
    cli.add_argument("--use-recovery",action="store_true")
    args = cli.parse_args()
    if args.stage=="register":
        return register(args)
    manifest = read(args.out/"manifest.json")
    if sha_file(args.config)!=manifest["config_sha256"]:
        raise ValueError("changed official environment config")
    if args.stage=="ranker":
        from .alfworld_ranker import train
        paths = [args.out/"demos"/(s["id"]+".json") for s in manifest["splits"]["demonstration"]]
        if not all(p.exists() for p in paths):
            raise ValueError("expert collection is incomplete")
        return train(args.out,manifest["seed"],args.rich_ranker,args.lexical_ranker,args.use_recovery)
    if args.stage=="recovery" and args.recovery_ranker is None:
        cli.error("recovery needs --recovery-ranker")
    corpus(args,manifest)


if __name__=="__main__":
    main()
