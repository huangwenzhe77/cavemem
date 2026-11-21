"""Terminal-free mini live evaluation and faithful Harbor system-comparison entry points."""
import argparse
import asyncio
import hashlib
import shutil
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from . import formal, formal_plan as fp, gate_validation as gv, mini, prospective
from .constrained_gate import Policy
from .gate import signed
from .models import FrozenQwen, xml_parser
from .records import digest, read, save
from .replay import require
from .runtime_pair import uniform
from .single_arm import Episode, durable


SYSTEMS = ("silent", "always_remind", "local_checker", "policy")
COMPONENTS = ("fixed_relevance", "context_guideline", "query_bound_note", "contract_guard")
GUARDED = "guarded_dr"


def query_note(card_text, task, working):
    return (f"Reusable procedure: {card_text}\n"
            f"Target binding: the active list has {len(working)} entries; the requested order is {task['family']}.\n"
            "Applicability: the requested operation keeps the complete input multiset. "
            "Verification: before submitting, check the order and retain each input occurrence.")


def local_check(task, state, action):
    """Dry-run the mini DSL against the public sorting contract; consumes no hidden grader signal."""
    after = deepcopy(state)
    mini.execute(after, action["commands"])
    expected = sorted(task["input"], reverse=task["family"]=="descending")
    if after["submitted"] is not None:
        return after["submitted"] != expected
    return bool(action["complete"] or Counter(after["working"]) != Counter(expected))


class LiveEpisode(Episode):
    def select(self, folder, pre, draft):
        self.checkpoint = pre["checkpoint_id"]
        self.selected_turn = pre["step"]
        task = self.row["task"]
        start = time.perf_counter()
        if task["family"] not in self.cards:
            self.main_status = "no_source_card_for_mechanism"
            self.usage["matching_seconds"] += time.perf_counter() - start
            durable(folder/"selection.json", {"status":self.main_status,"a0_digest":draft["a0_digest"]})
            return self.messages, None
        route = gv.route if task["template_id"] in gv.TEMPLATES else fp.formal_route
        decision = route(self.cards, task, pre, draft)
        self.usage["matching_seconds"] += time.perf_counter() - start
        if not decision.get("memory_id") or (self.system != "fixed_relevance" and decision.get("rho") != 1):
            self.main_status = decision["status"]
            durable(folder/"selection.json", {"status":self.main_status,"decision":decision})
            return self.messages, None
        messages = deepcopy(self.messages)
        note = decision["reminder"]
        if self.system == "query_bound_note":
            note = query_note(note,task,pre["state"]["working"])
        messages[-1]["content"] += "\n\nAdvisory note for this proposal only: " + note
        count = self.prompt_tokens(messages)
        if not self.fits(count) or self.usage["tool_calls"] + len(draft["a0"]["commands"]) > self.plan["total_tools"]:
            self.main_status = "insufficient_budget"
            durable(folder/"selection.json", {"status":self.main_status,"decision":decision})
            return self.messages, None
        start = time.perf_counter()
        x = formal.gate_inputs(self.row,pre,draft,decision,self.plan)
        if self.system == GUARDED:
            result = self.dr_policy.decide(x,uniform("transfer-policy",self.row["seed"],pre["step"]),
                                           head="dr", guarded=True)
        elif self.system == "policy":
            result = self.policy.decide(x,uniform("transfer-policy",self.row["seed"],pre["step"]))
        elif self.system == "local_checker":
            result = {"remind":local_check(task,pre["state"],draft["a0"])}
        elif self.system == "contract_guard":
            result = {"remind":decision["v"] == decision["rho"] == decision["e"] == 1}
        else:
            result = {"remind":self.system in ("always_remind","context_guideline","query_bound_note","fixed_relevance")}
        self.usage["assignment_seconds"] += time.perf_counter() - start
        self.main_status = "decided"
        self.assigned = {**result, "arm":"remind" if result["remind"] else "silent",
                         "system":self.system, "checkpoint_id":self.checkpoint, "features":x, "decision":decision,
                         "candidate_note":note}
        durable(folder/"selection.json", self.assigned)
        return messages, count


def weight_signature(path):
    weights = sorted(Path(path).glob("*.safetensors"))
    require(weights, "missing local safetensor weights")
    keys = []
    for p in weights:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda:f.read(8*1024*1024), b""):
                h.update(block)
        keys.append((p.name,h.hexdigest()))
    return digest(keys)


def tasks(args, core):
    if args.split == "sealed":
        doc = read(core/"sealed-test.json")
        require(digest(doc)==signed(core/"preregistration.json","plan_id")["test_manifest_digest"], "test seal changed")
        return doc["tasks"]
    # Known development examples: wiring and mechanism coverage only.
    reg = signed(Path("outputs/cave-runtime-pair-01/preregistration.json"), "plan_id")
    cohort = reg["cohorts"][0]["tasks"]
    rows = []
    for template in ("batch-down","list-up"):
        r = next(x for x in cohort if x["task"]["template_id"]==template)
        rows.append({**r,"main_opportunity":1,"q":1.})
    for n, split in enumerate(("target_val","target_test")):
        task = {**mini.task(split, 73101+n),"template_id":"external-mechanism-"+split}
        rows.append({"episode_id":f"ood-{n}","task":task,"task_digest":digest(task),"seed":73101+n,
                     "main_opportunity":1,"q":1.})
    return rows


def live(args):
    core = Path("outputs/cave-formal-01")
    plan = signed(core/"preregistration.json","plan_id")
    rows = tasks(args,core)
    systems = tuple(args.systems)
    require(len(set(systems)) == len(systems), "repeated system")
    base = Path(plan["actor"]["actor"])
    actor_paths = [base] + ([args.second_actor] if args.second_actor else [])
    signatures = [weight_signature(path) for path in actor_paths]
    require(len(set(signatures))==len(signatures), "two distinct frozen weight sets required")
    modelfile = Path("outputs/cave-primal-dual-01/model.json")
    policy = Policy(modelfile,Path("outputs/cave-mlp-01/frozen/model.json"))
    dr_policy = None
    if GUARDED in systems:
        from .logged_policy_fit import Runtime
        dr_policy = Runtime(args.dr_model,Path("outputs/cave-mlp-01/frozen/model.json"))
    cards = prospective.verified_cards(plan,fp.DEFAULT_BANK,fp.DEFAULT_PACKAGE,Path("outputs/cave-pilot-01/preregistration.json"))
    args.out.mkdir(parents=True,exist_ok=False)
    manifest = {"split":args.split,"policy_id":policy.doc["policy_id"],"code":fp.code_digest("transfer_suite.py"),
                "single_arm_code":fp.code_digest("single_arm.py"), "source_plan_id":plan["plan_id"],"task_ids":[r["task_digest"] for r in rows],
                "systems":systems,"dr_model_id":dr_policy.doc["id"] if dr_policy else None,
                "actor_weight_ids":signatures,"max_steps":plan["max_steps"],
                "max_new_tokens":256,"total_tokens":plan["total_tokens"],"total_tools":plan["total_tools"]}
    pid=digest(manifest)
    durable(args.out/"registration.json",{**manifest,"study_id":pid})
    grouped=defaultdict(list)
    for a,path in enumerate(actor_paths):
        actor=FrozenQwen(path,256,0)
        if a==0:
            require(actor.info==plan["actor"],"primary actor drift")
        settings={**plan,"actor":actor.info}
        durable(args.out/f"actor-{a}/model.json", actor.info)
        for i,row in enumerate(rows):
            # Rotate system order; all systems receive identical task, draft seed and intervention schedule.
            order=systems[i%len(systems):]+systems[:i%len(systems)]
            for system in order:
                dest=args.out/f"actor-{a}"/system/row["episode_id"]
                ep=LiveEpisode(dest,settings,pid,pid,row,actor,xml_parser(),cards,None)
                ep.system,ep.policy=system,policy
                ep.dr_policy=dr_policy
                res=ep.run()
                durable(dest/"result.json",res)
                value={"actor":a,"system":system,"task_id":row["task_digest"],"mechanism":row["task"]["family"],
                       "eligible":res["assignment"] is not None,"arm":res["assigned_arm"],"reward":res["verifier"]["reward"],
                       "utility":res["usage"]["observed_utility"],"tokens":res["usage"]["total_tokens"],
                       "seconds":res["usage"]["episode_wall_seconds"],"status":res["main_status"]}
                grouped[f"actor-{a}/{system}"].append(value)
        del ep, actor
    save(args.out/"results.json",dict(grouped))
    print(args.out/"results.json")


def harbor(args):
    from ..config import load_config
    from ..data import create_adapter
    from ..runner import run_evaluation
    cfg=load_config(args.config)
    cfg.run.task_names=[args.task]
    cfg.run.tasks_dir=str(args.cache)
    cfg.enroot.sqsh_dir=str(args.images)
    cfg.run.harbor_trusted_grading=True
    cfg.run.cave_capture=False
    if args.system=="bare":
        cfg.memory.enabled=False
    elif args.system=="pma":
        cfg.memory.enabled=True
    require(cfg.run.adapter_type=="harbor","Harbor task cache required")
    adapter=create_adapter(adapter_type="harbor",tasks_dir=str(args.cache),sqsh_prefix=cfg.run.harbor_sqsh_prefix)
    found=adapter.load_tasks(task_names=[args.task]) if args.cache.exists() else []
    images=adapter.load_tasks(task_names=[args.task],sqsh_dir=str(args.images)) if found and args.images.is_dir() else []
    status={"system":args.system,"agent_class":"MemoryEnabledTerminus2" if cfg.memory.enabled else "Terminus2",
            "actor":cfg.model.model_name,"memory_model":cfg.memory.model_name if cfg.memory.enabled else None,
            "max_turns":cfg.model.max_turns,"task":args.task,"cached_task":len(found)==1,
            "matching_sqsh":len(images)==1,"enroot":shutil.which("enroot"),"trusted_grading":True,
            "launch_requested":args.launch,"actor_endpoint_verified":False}
    args.out.mkdir(parents=True,exist_ok=False)
    save(args.out/"preflight.json",status)
    print(status,flush=True)
    if args.launch:
        require(found and images and status["enroot"],"backend prerequisites missing (see preflight.json)")
        cfg.run.output_dir=str(args.out)
        cfg.run.n_parallel=1
        cfg.run.resume_run_dir=None
        return asyncio.run(run_evaluation(cfg,run_id="system-run"))


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    sub=cli.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("mini")
    p.add_argument("--split",choices=["development","sealed"],required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--second-actor",type=Path)
    p.add_argument("--systems",nargs="+",choices=SYSTEMS+COMPONENTS+(GUARDED,),default=SYSTEMS+COMPONENTS)
    p.add_argument("--dr-model",type=Path,default=Path("outputs/cave-comparison-01/model.json"))
    h=sub.add_parser("harbor")
    h.add_argument("--system",choices=["bare","pma"],required=True)
    h.add_argument("--config",type=Path,required=True)
    h.add_argument("--task",default="regex-log")
    h.add_argument("--cache",type=Path,default=Path.home()/".cache/harbor/tasks")
    h.add_argument("--images",type=Path,default=Path("/root/autodl-tmp/cave-mem/sqsh"))
    h.add_argument("--out",type=Path,required=True)
    h.add_argument("--launch",action="store_true")
    args=cli.parse_args()
    live(args) if args.cmd=="mini" else harbor(args)

