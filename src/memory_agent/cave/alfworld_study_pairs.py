"""Validate transferable memories and measure matched policy outcomes on whole splits."""

import contextlib
import io
import os
import random
from collections import Counter
from statistics import mean, median
from math import exp, isfinite
from time import monotonic

import torch

from .alfworld_env import Episode, MAX_STEPS, sha_file
from .alfworld_experiment import costs
from .alfworld_ranker import entities,shape,words
from .alfworld_study_rollout import receipt, rollout
from .models import TeacherCallError, teacher_request
from .records import digest, read, save


FUNCTION_WORDS = frozenset("a an the some it and on in at to with under of then".split())


def guidance(anchor, correction, variant, *, abstract=False):
    source = {"goal":anchor["goal"],"visible_observation":anchor["observation"],
              "recent_actions":[r["draft"]["action"] for r in anchor["prefix"][-5:]],
              "original":anchor["draft"]["action"],"corrected":correction}
    prompt = ("In this historical game, the original cached action led to failure and the alternative "
              "was independently replayed to a win. Explain how the next decision should use the visible "
              "evidence. Output a JSON object with reminder, a short transferable conditional instruction. "
              "Avoid entity IDs, facts outside this input, or instructions to copy a past solution. ")
    prompt += ("Use at most 35 words and advise a directly executable next action." if variant==0 else
               "Use at most 18 words. If the intended action is already available on the menu, state the general "
               "reason to use it before repeating the earlier behavior.")
    if abstract:
        prompt += (" Do not use names of objects, containers, or locations from this source game. "
                   "Refer to public roles such as the requested item, the goal destination or the visible surface. "
                   "Describe exactly when this immediate action applies in another instance.")
    def validate(_, card):
        if not isinstance(card.get("reminder"),str) or not 1<=len(card["reminder"].split())<=50:
            raise ValueError("expected short lesson")
        return {"reminder":card["reminder"]}
    for style in ("chat","responses"):
        try:
            return teacher_request(source,os.environ["OPENAI_BASE_URL"],"gpt-6-sol",style,
                                   instruction=prompt,validate=validate)
        except TeacherCallError:
            if style=="responses":
                raise


def anchor_at(rows,i):
    return {**rows[i],"prefix":rows[:i]}


def source_candidates(natural, cap, actor):
    rows = natural["trace"]["steps"]
    if natural["metrics"]["success"] and len(rows)<=cap:
        return
    candidates = []
    history = []
    for i,r in enumerate(rows[:cap-2]):
        original = r["draft"]["action"]
        suggestion = r["draft"]["ranked_candidates"][0]
        recorded = list(history)
        history.append(original)
        if original == suggestion or suggestion not in r["draft"]["allowed"]:
            continue
        menu = r.get("public_menu",r["draft"]["allowed"])
        matrix = actor.ranker.features(r["goal"],r["observation"],recorded,menu)
        with torch.inference_mode():
            values = actor.ranker(matrix)
            margin = (values[menu.index(suggestion)]-values[menu.index(original)]).item()
        candidates.append((margin,i,suggestion))
    grouped = set()
    for margin,i,suggestion in sorted(candidates,key=lambda z:(-z[0],z[1])):
        key = shape(rows[i]["draft"]["action"]),shape(suggestion)
        if key in grouped:
            continue
        grouped.add(key)
        yield anchor_at(rows,i),suggestion
        if len(grouped)==3:
            break


def audit_prefix(args, full, cap):
    trace = full["trace"]
    extra = full["extra_draft"]
    rows = trace["steps"][:cap-int(extra is not None)]
    if len(rows)==len(trace["steps"]):
        return full["metrics"]
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        env = Episode(args.data,args.config,full["spec"])
        try:
            clone = env.restore([r["execution"] for r in rows],rows[-1]["execution"]["state"])
            try:
                prefix = {"success":int(clone.won),"steps":rows}
                return costs(prefix) if extra is None else costs(prefix,0,extra)
            finally:
                clone.env.close()
        finally:
            env.env.close()


def embedding(actor,anchor,action):
    menu = anchor.get("public_menu",anchor["draft"]["allowed"])
    features = actor.ranker.features(anchor["goal"],anchor["observation"],
                                     [r["draft"]["action"] for r in anchor["prefix"]],menu)
    ix = menu.index(action)
    with torch.inference_mode():
        return torch.nn.functional.normalize(actor.ranker.net[:-1](features[ix:ix+1]),dim=-1)[0].tolist()


def cached(path, callback):
    if path.exists():
        return read(path)
    result = callback()
    save(path,result)
    return result


def rendered(note, action, grounded):
    if not grounded:
        return note
    return (note + f" The memory matcher found this currently legal candidate: {action}. "
            "Consider choosing it now if the evidence and the task support it.")


def create_memories(args,manifest,actor,*,grounded=False):
    specs = manifest["splits"]["historical"]
    dev = [read(receipt(args.out,s)) for s in manifest["splits"]["development"]]
    lengths = sorted(r["metrics"]["steps"] for r in dev if r["metrics"]["success"])
    if len(lengths)<len(dev)*.25:
        raise RuntimeError("development success too low to evaluate interference")
    budget = lengths[max(0,int(.8*len(lengths))-1)]
    plan = {"budget":budget,"selection":"smallest development horizon retaining 80% of full successes, rounded down by one empirical count",
            "scopes": [MAX_STEPS,budget],"max_direct_screens":20,"max_cards":6,
            "source_admission":"two fresh deterministic replays per arm and reorder, silent fail and both alternatives win",
            "screen_order":"historical cases that barely miss the fixed budget first, then full-horizon failures",
            "grounded_legal_candidate_in_prompt":grounded}
    if grounded and (args.out/"test-freeze.json").exists():
        raise ValueError("cannot change intervention after release")
    gate = args.out/("source-plan-grounded.json" if grounded else "source-plan.json")
    if gate.exists() and read(gate)!=plan:
        raise ValueError("source plan changed")
    save(gate,plan)
    bank, direct_total, direct_wins = [], 0, 0
    observations = [(s,read(receipt(args.out,s))) for s in specs]
    observations.sort(key=lambda item:(not bool(item[1]["metrics"]["success"]),
                                        abs(item[1]["metrics"]["steps"]-budget),item[0]["id"]))
    for spec,n in observations:
        for cap in sorted(set(plan["scopes"]),reverse=True):
            for anchor,action in source_candidates(n,cap,actor):
                step = len(anchor["prefix"])+1
                dest = receipt(args.out,spec).parent/f"cap-{cap}-step-{step}"
                name = dest/"direct.json"
                direct = cached(name,lambda: rollout(args,spec,actor,anchor=anchor,correction=action,cap=cap))
                direct_total += 1
                direct_wins += direct["metrics"]["success"]
                print("DIRECT",spec["id"],cap,step,direct["metrics"]["success"],flush=True)
                if direct["metrics"]["success"]:
                    for variant in range(2):
                        proposer = cached(dest/f"teacher-{variant}.json",lambda:guidance(anchor,action,variant))
                        note = proposer["candidate"]["reminder"]
                        stem = f"{variant}-grounded" if grounded else str(variant)
                        wording = rendered(note,action,grounded)
                        text = cached(dest/f"reminder-screen-{stem}.json",
                                      lambda:rollout(args,spec,actor,anchor=anchor,reminder=wording,cap=cap))
                        if not text["metrics"]["success"]:
                            print("TEXT_FAIL",spec["id"],cap,step,variant,flush=True)
                            continue
                        reps = []
                        for i,order in enumerate((("silent","direct","remind"),("remind","direct","silent")),1):
                            arms = {}
                            for arm in order:
                                r = cached(dest/f"{arm}-{stem}-{i}.json",lambda: rollout(
                                    args,spec,actor,anchor=anchor,reminder=wording if arm=="remind" else None,
                                    correction=action if arm=="direct" else None,cap=cap))
                                arms[arm] = r["metrics"]
                            reps.append(arms)
                            if [arms[a]["success"] for a in ("silent","direct","remind")]!=[0,1,1]:
                                break
                        if len(reps)!=2 or any([a[z]["success"] for z in ("silent","direct","remind")]!=[0,1,1] for a in reps):
                            continue
                        gains = [a["remind"]["utility"]-a["silent"]["utility"] for a in reps]
                        if min(gains)<=0:
                            continue
                        path = dest/f"accepted-{stem}.json"
                        save(path,{"source":spec["id"],"anchor":anchor,"action":action,"reminder":wording,"metrics":reps})
                        bank.append({"id":digest([spec,cap,anchor["state"],wording]),"source":spec["id"],"text":note,
                                     "evidence":str(path.relative_to(args.out)), "sha256":sha_file(path),
                                     "shape":shape(action), "vector":embedding(actor,anchor,action),
                                     "utility":mean(gains),"grounded":grounded,
                                     "direct_remaining":int(median([r["direct"]["steps"]-step+1 for r in reps]))})
                        print("ADMIT",spec["id"],len(bank),flush=True)
                        break
                if direct_total>=plan["max_direct_screens"] or len(bank)>=plan["max_cards"] or (direct_total>=6 and not direct_wins):
                    break
            if direct_total>=plan["max_direct_screens"] or len(bank)>=plan["max_cards"] or (direct_total>=6 and not direct_wins):
                break
        if direct_total>=plan["max_direct_screens"] or len(bank)>=plan["max_cards"] or (direct_total>=6 and not direct_wins):
            break
    if grounded and (args.out/"bank.json").exists() and not (args.out/"bank-ungrounded.json").exists():
        save(args.out/"bank-ungrounded.json",read(args.out/"bank.json"))
    save(args.out/"bank.json",{"cards":bank,"direct_screens":direct_total,"direct_wins":direct_wins,"plan":plan})
    if not bank:
        raise RuntimeError("no source-verified textual memory; stop before evaluation")


def retrieve(actor,anchor,bank,threshold):
    result = []
    for a in anchor["draft"]["ranked_candidates"]:
        cards = [c for c in bank if c["shape"]==shape(a) and context_applies(actor,anchor,a,c)]
        if not cards:
            continue
        x = embedding(actor,anchor,a)
        for c in cards:
            score = sum(u*v for u,v in zip(x,c["vector"],strict=True))
            if score>=threshold:
                result.append((score,c["utility"],c["id"],a,c))
    # Relevance first defines eligibility. Source utility ranks the eligible records.
    return max(result,key=lambda s:(s[1],s[0],s[2])) if result else None


def contextual_contract(actor,anchor,action):
    vocabulary = actor.ranker.vocabulary
    roles = list(dict.fromkeys(w for w in words(anchor["goal"]) if w in vocabulary))
    actions = [shape(r["draft"]["action"]) for r in anchor["prefix"]]
    es = entities(action)
    mentioned = entities(anchor["observation"])
    visible = set(mentioned)
    changed = []
    for r in anchor["prefix"]:
        executed = r["execution"]["action"]
        feedback = words(r["execution"]["observation"])
        verb = words(executed)[0]
        if len(feedback)>1 and feedback[0]=="you" and feedback[1].startswith(verb):
            changed.append((shape(executed),set(entities(executed))))
    arg_progress = [sorted({kind for kind,args in changed if e in args}) for e in es]
    return {"argument_roles":[roles.index(name) if name in roles else -1 for name,_ in es],
            "arguments_visible":[e in visible for e in es],
            "prior":actions[-3:],
            "goal_words":sorted(set(words(anchor["goal"]))-vocabulary-FUNCTION_WORDS),
            "argument_progress":arg_progress,
            "visible_goal_roles":[any(n==r for n,_ in visible) for r in roles],
            "first_visible_goal_role":roles.index(mentioned[0][0]) if mentioned and mentioned[0][0] in roles else -1,
            "observed_empty":"you see nothing" in anchor["observation"].lower()}


def context_applies(actor,anchor,action,card):
    expected = card.get("contract")
    if expected is None:
        return True
    current = contextual_contract(actor,anchor,action)
    if (current["argument_roles"]!=expected["argument_roles"] or
            current["arguments_visible"]!=expected["arguments_visible"]):
        return False
    if bool(current["prior"])!=bool(expected["prior"]):
        return False
    if expected["prior"] and expected["prior"][-1] not in current["prior"]:
        return False
    if expected["goal_words"]!=current["goal_words"]:
        return False
    if (expected.get("visible_goal_roles")!=current["visible_goal_roles"] or
            expected.get("first_visible_goal_role")!=current["first_visible_goal_role"]):
        return False
    if "empty" in words(card["text"]) and not current["observed_empty"]:
        return False
    return all(set(prior).issubset(now) for prior,now in
               zip(expected.get("argument_progress",current["argument_progress"]),
                   current["argument_progress"],strict=True))


def choose(actor,rows,bank,threshold):
    for i in range(min(len(rows),MAX_STEPS-1)):
        anchor = anchor_at(rows,i)
        hit = retrieve(actor,anchor,bank,threshold)
        if hit is not None:
            return anchor,hit
    return None


def bank_checked(args):
    doc = read(args.out/"bank.json")
    if not doc["cards"]:
        raise ValueError("empty memory bank")
    for c in doc["cards"]:
        path = (args.out/c["evidence"]).resolve()
        path.relative_to(args.out.resolve())
        if sha_file(path)!=c["sha256"]:
            raise ValueError("changed source evidence")
    return doc


def tighten_contracts(args,actor):
    if (args.out/"test-freeze.json").exists():
        raise ValueError("official evaluation already released")
    doc = bank_checked(args)
    path = args.out/"bank-before-role-guard.json"
    if not path.exists():
        save(path,doc)
    for card in doc["cards"]:
        evidence = read(args.out/card["evidence"])
        card["contract"] = contextual_contract(actor,evidence["anchor"],evidence["action"])
        if not context_applies(actor,evidence["anchor"],evidence["action"],card):
            # For example, a word-conditioned empty clause was never observed in the actual source.
            card["ineligible_source_contract"] = True
    doc["cards"] = [c for c in doc["cards"] if not c.get("ineligible_source_contract")]
    doc["contract_protocol"] = "normalized goal modifiers, observed legal argument alignment and confirmed history"
    save(args.out/"bank.json",doc)
    print("SEMANTIC_BANK",len(doc["cards"]),flush=True)


def diverse_source_order(choices):
    """Alternate earlier and later states instead of exhausting near-terminal corrections first."""
    by_progress = {}
    for choice in choices:
        _,_,neg_i,_,spec,natural = choice
        rows = natural["trace"]["steps"]
        remaining = len(rows)+neg_i
        if remaining<2:
            continue
        key = spec["family"],min(2,3*remaining//len(rows))
        by_progress.setdefault(key,[]).append(choice)
    pools = [sorted(rows,key=lambda r:r[:3],reverse=True) for rows in by_progress.values()]
    while pools:
        for pool in pools:
            yield pool.pop()
        pools = [pool for pool in pools if pool]


def historical_exploration(args,manifest,actor,*,abstract=False,diverse=False):
    """Recover evidence from actual, explicitly exploratory actor decisions in training games."""
    out = args.out
    if (out/"test-freeze.json").exists():
        raise ValueError("cannot change source evidence after the official cohort was released")
    prior = read(out/"bank.json")
    if prior["cards"] and not abstract:
        return
    label = "diverse" if diverse else "abstract" if abstract else "exploration"
    backup = "abstraction" if label=="abstract" else label
    save(out/f"bank-before-{backup}.json",prior)
    successful = [(s,read(receipt(out,s))) for s in manifest["splits"]["historical"]]
    successful = [(s,n) for s,n in successful if n["metrics"]["success"]]
    dev = sorted(r["metrics"]["steps"] for s in manifest["splits"]["development"]
                 if (r:=read(receipt(out,s)))["metrics"]["success"])
    budgets = sorted({prior["plan"]["budget"], MAX_STEPS,
                      dev[max(0,int(.60*len(dev))-1)]})
    protocol = {"budget_candidates":budgets,"target_budget":prior["plan"]["budget"],
                "input_split":"historical (official train)", "max_samples":48 if diverse else 36 if abstract else 16,
                "max_cards":8 if diverse else 6 if abstract else 4,
                "draw":"1.2-temperature distribution over legal non-greedy choices from fixed actor logits",
                "source_step_pool":("round robin across task families and relative progress thirds, excluding the last decision"
                                    if diverse else "last eight decisions of a training success with at least one spare call"),
                "source_proof":"two independently restored deterministic futures for each cached sampled anchor",
                "one_card_per_source_game":abstract,"role_contract":abstract}
    plan_file = "exploration-plan.json" if label=="exploration" else f"{label}-exploration-plan.json"
    save(out/plan_file,protocol)
    choices = []
    for s,n in successful:
        rows = n["trace"]["steps"]
        after = next((b for b in budgets if b>=len(rows)+1),None)
        if after is None:
            continue
        for i in range(0 if diverse else max(0,len(rows)-8),len(rows)):
            draft = rows[i]["draft"]
            allowed = draft["allowed"]
            if len(allowed)<2:
                continue
            choices.append((after-len(rows),s["id"],-i,after,s,n))
    cards, samples, noisy_failures, direct_wins, text_failures = [],0,0,0,0
    names_from_source = {p["id"]:set(words(n["trace"]["steps"][0]["goal"])) & actor.ranker.vocabulary
                         for p,n in successful}
    queue = diverse_source_order(choices) if diverse else sorted(choices,key=lambda q:q[:3])
    for _,_,neg_i,cap,s,n in queue:
        if abstract and any(c["source"]==s["id"] for c in cards):
            continue
        i=-neg_i
        good = anchor_at(n["trace"]["steps"],i)
        draft = good["draft"]
        expected = draft["action"]
        if "choice_logits" not in draft:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                ep = Episode(args.data,args.config,s)
                try:
                    restored = ep.restore([r["execution"] for r in good["prefix"]],good["state"])
                    try:
                        regenerated = actor.propose(restored)
                    finally:
                        restored.env.close()
                finally:
                    ep.env.close()
            if any(regenerated[k]!=draft[k] for k in ("action","allowed","prompt_digest")):
                raise ValueError("historical draft changed during source exploration")
            draft = regenerated
        if not all(isfinite(x) for x in draft["choice_logits"]):
            raise ValueError("invalid historical label logits")
        options = [(a,l/1.2) for a,l in zip(draft["allowed"],draft["choice_logits"],strict=True) if a!=expected]
        weights = [exp(l-max(z[1] for z in options)) for _,l in options]
        seed = int(digest([manifest["seed"],s["id"],i,"source-exploration"])[:12],16)
        proposal, probability = random.Random(seed).choices(
            [(o[0],w/sum(weights)) for o,w in zip(options,weights,strict=True)],weights,k=1)[0]
        anchor = {**good,"draft":{**draft,"action":proposal,"source_draw":seed,"source_probability":probability}}
        dest = receipt(out,s).parent/f"{label}-{cap}-step-{i+1}"
        save(dest/"anchor.json",anchor)
        silent = cached(dest/"silent-screen.json",lambda:rollout(args,s,actor,anchor=anchor,cap=cap))
        samples += 1
        if silent["metrics"]["success"]:
            print("SOURCE_EXPLORE",s["id"],i+1,"silent still wins",flush=True)
        else:
            noisy_failures += 1
            direct = cached(dest/"direct-screen.json",lambda:rollout(
                args,s,actor,anchor=anchor,correction=expected,cap=cap))
            print("SOURCE_EXPLORE",s["id"],i+1,"silent failed; direct",direct["metrics"]["success"],flush=True)
            if direct["metrics"]["success"]:
                direct_wins += 1
                for variant in range(2):
                    teacher = cached(dest/f"teacher-{variant}.json",lambda:guidance(
                        anchor,expected,variant,abstract=abstract))
                    note = teacher["candidate"]["reminder"]
                    if abstract and set(words(note)) & names_from_source[s["id"]]:
                        print("ABSTRACTION_SOURCE_NAMES",s["id"],i+1,variant,flush=True)
                        continue
                    wording = rendered(note,expected,True)
                    trial = cached(dest/f"text-screen-{variant}.json",lambda:rollout(
                        args,s,actor,anchor=anchor,reminder=wording,cap=cap))
                    if not trial["metrics"]["success"]:
                        text_failures += 1
                        print("EXPLORATION_TEXT_FAIL",s["id"],i+1,variant,flush=True)
                        continue
                    reps = []
                    for rep,order in enumerate((("silent","direct","remind"),("remind","direct","silent")),1):
                        observed = {}
                        for a in order:
                            result = cached(dest/f"{a}-{variant}-{rep}.json",lambda:rollout(
                                args,s,actor,anchor=anchor,cap=cap,
                                reminder=wording if a=="remind" else None,
                                correction=expected if a=="direct" else None))
                            observed[a] = result["metrics"]
                        reps.append(observed)
                    if any([r[a]["success"] for a in ("silent","direct","remind")]!=[0,1,1] for r in reps):
                        continue
                    gains = [r["remind"]["utility"]-r["silent"]["utility"] for r in reps]
                    if min(gains)<=0:
                        continue
                    evidence = dest/f"accepted-{variant}.json"
                    save(evidence,{"source":s["id"], "anchor":anchor,"action":expected,
                                   "reminder":wording,"metrics":reps,"protocol":protocol})
                    cards.append({"id":digest([s,cap,anchor["state"],wording]),"source":s["id"],"text":note,
                                  "evidence":str(evidence.relative_to(out)),"sha256":sha_file(evidence),
                                  "shape":shape(expected),"vector":embedding(actor,anchor,expected),
                                  "utility":mean(gains),"grounded":True,
                                  "direct_remaining":int(median([r["direct"]["steps"]-i for r in reps]))})
                    if abstract:
                        cards[-1]["contract"] = contextual_contract(actor,anchor,expected)
                    print("EXPLORATION_ADMIT",s["id"],len(cards),flush=True)
                    break
        if len(cards)>=protocol["max_cards"] or samples>=protocol["max_samples"]:
            break
        if diverse and not cards and ((noisy_failures>=12 and not direct_wins) or text_failures>=12):
            print("SOURCE_EXPLORE_STOP",noisy_failures,direct_wins,text_failures,flush=True)
            break
    bank = {"cards":cards,"direct_screens":noisy_failures,"exploratory_drafts":samples,
            "plan":prior["plan"],"exploration":protocol}
    if diverse:
        bank.update(direct_wins=direct_wins,text_failures=text_failures)
        save(out/"diverse-source-report.json",bank)
        if not cards:
            raise RuntimeError("diverse source exploration did not reproduce a rescue; keep the prior bank")
        represented = {c["source"] for c in cards}
        bank = {**bank,"cards":[*cards,*(c for c in prior["cards"] if c["source"] not in represented)]}
    save(out/"bank.json",bank)
    if not cards:
        raise RuntimeError("exploratory source memories did not reproduce a rescue")


def grade_pair(args,spec,actor,natural,bank,threshold,cap,local_budget,*,baseline=None):
    t = monotonic()
    pick = choose(actor,natural["trace"]["steps"],bank,threshold)
    matching_time = monotonic()-t
    output = {"spec":spec["id"],"match":None,"scores":[],"matching_seconds":matching_time}
    limits = [(n,str(n)) for n in sorted(set([cap,MAX_STEPS]))]
    if pick:
        anchor,hit = pick
        step = len(anchor["prefix"])+1
        output["match"] = {"step":step,"score":hit[0],"action":hit[3],"card":hit[4]["id"],
                           "original":anchor["draft"]["action"],
                           "aligned":anchor["draft"]["action"]==hit[3]}
        holder = receipt(args.out,spec).parent
        save(holder/"selected.json",{"checkpoint":anchor,"match":output["match"]})
        # For an exact deterministic silent continuation, validate the actual trace
        # later through environment replay at the chosen budget.
        note = rendered(hit[4]["text"],hit[3],hit[4].get("grounded",False))
        name = "remind-" + digest([anchor["state"],hit[4]["id"],note])[:16]+".json"
        remind = cached(holder/name,lambda:rollout(args,spec,actor,anchor=anchor,reminder=note))
        output["match"]["same_revision"] = remind["trace"]["steps"][step-1]["draft"]["action"]==anchor["draft"]["action"]
    else:
        step = MAX_STEPS
    limits.append((min(MAX_STEPS,step-1+local_budget) if pick else MAX_STEPS,
                   f"checkpoint:{local_budget}"))
    if args.source_relative_budget:
        source_cap = min(MAX_STEPS,step-1+max(2,hit[4]["direct_remaining"])) if pick else MAX_STEPS
        limits.append((source_cap,"source-relative"))
    measured = {}
    for limit,setting in limits:
        if limit not in measured:
            s = baseline.get(limit) if baseline is not None else None
            if s is None:
                s = audit_prefix(args,natural,limit)
            r = audit_prefix(args,remind,limit) if pick and step<limit else dict(s)
            measured[limit] = {"cap":limit,"silent":s,"remind":r,"eligible":bool(pick and step<limit)}
        output["scores"].append({**measured[limit],"setting":setting})
    return output


def aggregate(records):
    tables = {}
    for item in records:
        for r in item["scores"]:
            z = tables.setdefault(r["setting"],[])
            z.append(({**item,"scores":None},r))
    report = {}
    for cap,group in tables.items():
        metrics, aligned = Counter(), Counter()
        cases = []
        for item,r in group:
            s,m=r["silent"],r["remind"]
            metrics["silent_success"]+=s["success"]
            metrics["remind_success"]+=m["success"]
            metrics["eligible"]+=int(r["eligible"])
            ds,dt,dsec=m["success"]-s["success"],m["tokens"]-s["tokens"],m["seconds"]-s["seconds"]
            metrics["harm"]+=int(ds<0)
            metrics["rescue"]+=int(ds>0)
            metrics["tokens_delta"]+=dt
            metrics["seconds_delta"]+=dsec
            metrics["dual_success_more"]+=int(s["success"] and m["success"] and dt>0 and dsec>0)
            metrics["harm_more"]+=int(ds<0 and dt>0 and dsec>0)
            if ds<0:
                cases.append({"id":item["spec"],"tokens_delta":dt,"seconds_delta":dsec,"match":item["match"]})
            fired = bool(r["eligible"] and item["match"]["aligned"])
            aligned["reminders"] += int(fired)
            aligned["harm"] += int(fired and ds<0)
            aligned["rescue"] += int(fired and ds>0)
            aligned["remind_success"] += m["success"] if fired else s["success"]
            aligned["tokens_delta"] += dt if fired else 0
            aligned["seconds_delta"] += dsec if fired else 0
        report[str(cap)]={"tasks":len(group),**dict(metrics),"harm_cases":cases,
                          "aligned_only_policy":dict(aligned)}
    return report


def pilot(args,manifest,actor):
    tighten_contracts(args,actor)
    bank = bank_checked(args)
    rows = [(s,read(receipt(args.out,s))) for s in manifest["splits"]["development"]]
    calibration = {}
    for tau in (.96,.92,.88,.84,.80):
        hits = [choose(actor,n["trace"]["steps"],bank["cards"],tau) for _,n in rows]
        calibration[str(tau)] = {"eligible":sum(h is not None for h in hits),
                                 "aligned":sum(h[0]["draft"]["action"]==h[1][3] for h in hits if h)}
    ready = [float(t) for t,r in calibration.items() if r["eligible"]>=6 and r["aligned"]>=2]
    viable = [float(t) for t,r in calibration.items() if r["eligible"]>=2]
    save(args.out/"retrieval-calibration.json",calibration)
    if not viable:
        raise RuntimeError("retrieval did not cover development episodes")
    threshold = max(ready or viable)
    local = max(2,int(median([m["direct_remaining"] for m in bank["cards"]])))
    records = []
    for s,n in rows:
        print("DEV_PAIR",s["id"],flush=True)
        records.append(grade_pair(args,s,actor,n,bank["cards"],threshold,bank["plan"]["budget"],local))
    metrics = aggregate(records)
    save(args.out/"dev-paired.json",metrics)
    cap = bank["plan"]["budget"]
    if metrics[str(MAX_STEPS)]["eligible"]<2:
        raise RuntimeError("source memory contract covers fewer than two development episodes")
    save(args.out/"test-freeze.json",{"cap":cap,"threshold":threshold,"bank":digest(bank),
                                    "local_budget":local,"local_rule":"median source direct steps to finish",
                                    "split_sizes":{"valid_seen":140,"valid_unseen":134},
                                    "no_match_policy":"silence","release":"source-verified reminders",
                                    "retrieval_calibration":calibration,
                                    "source_relative_budget":args.source_relative_budget})
    print("DEV_READY",metrics,flush=True)


def checked_baseline(args):
    root = args.baseline_from
    release = args.out/"baseline-run.json"
    if root is None:
        if release.exists():
            raise ValueError("resume with the original --baseline-from")
        return None
    if root.resolve()==args.out.resolve() or read(root/"audited.json")["n"]!=274:
        raise ValueError("baseline must be a separate audited benchmark run")
    inputs = ["manifest.json","actor.json","ranker.json","ranker.pt"]
    if args.examples:
        demos = sorted(p.relative_to(args.out) for p in (args.out/"demos").glob("*.json"))
        if demos!=sorted(p.relative_to(root) for p in (root/"demos").glob("*.json")):
            raise ValueError("baseline demonstration set changed")
        inputs.extend(demos)
    if any(sha_file(root/p)!=sha_file(args.out/p) for p in inputs):
        raise ValueError("baseline actor, demonstrations or benchmark changed")
    provenance = {"root":str(root.resolve()),"audit_sha256":sha_file(root/"audited.json")}
    if release.exists():
        if read(release)!=provenance:
            raise ValueError("baseline run changed")
    else:
        if any((args.out/split).exists() for split in ("valid_seen","valid_unseen")):
            raise ValueError("cannot add a reused baseline during evaluation")
        save(release,provenance)
    return root


def full(args,manifest,actor):
    bank,frozen = bank_checked(args),read(args.out/"test-freeze.json")
    if frozen["bank"]!=digest(bank):
        raise ValueError("frozen bank changed")
    if args.source_relative_budget!=frozen.get("source_relative_budget",False):
        raise ValueError("test budget settings differ from the development freeze")
    root = checked_baseline(args)
    groups = {}
    for group in ("valid_seen","valid_unseen"):
        rows = groups[group]=[]
        for spec in manifest["splits"][group]:
            path = receipt(args.out,spec)
            baseline = None
            if root is None:
                n = cached(path,lambda:rollout(args,spec,actor))
            else:
                prior = receipt(root,spec)
                n = cached(path,lambda:read(prior))
                if sha_file(path)!=sha_file(prior) or n["spec"]!=spec:
                    raise ValueError(f"baseline trajectory mismatch: {spec['id']}")
                paired = read(receipt(root,spec,"paired"))
                baseline = {r["cap"]:r["silent"] for r in paired["scores"]}
                if paired["spec"]!=spec["id"] or baseline[MAX_STEPS]!=n["metrics"]:
                    raise ValueError(f"baseline scores mismatch: {spec['id']}")
            result = cached(receipt(args.out,spec,"paired"),lambda:grade_pair(
                args,spec,actor,n,bank["cards"],frozen["threshold"],frozen["cap"],frozen["local_budget"],baseline=baseline))
            rows.append(result)
            print("TEST",group,len(rows),len(manifest["splits"][group]),
                  [(q["cap"],q["silent"]["success"],q["remind"]["success"]) for q in result["scores"]],flush=True)
            save(args.out/"progress.json",{name:aggregate(v) for name,v in groups.items()})
    save(args.out/"summary.json",{"groups":{k:aggregate(v) for k,v in groups.items()},
                                 "overall":aggregate([r for z in groups.values() for r in z]),"freeze":frozen})
