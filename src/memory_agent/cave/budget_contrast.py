"""Audit the budget manipulation and report matched effects per checkpoint."""
import argparse
import math
import random
from collections import Counter
from pathlib import Path

from .gate import lines, signed
from .records import digest, read, save
from .replay import require


def summarize(root):
    p=signed(root/"preregistration.json","plan_id")
    pairs=read(root/"paired-metrics.json")
    primary=read(root/"analysis.json")
    require(primary["sample_digest"]==digest(pairs) and primary["plan_id"]==p["plan_id"],"wrong budget dataset")
    bind=Counter()
    change=Counter()
    hist=Counter()
    differences=[]
    pair_cells=Counter()
    for pair in pairs:
        conditions=pair["by_budget"]
        if pair["D_one_minus_normal"] is None:
            continue
        differences.append(pair["D_one_minus_normal"])
        normal,one=conditions["normal"],conditions["one"]
        pair_cells[(normal["cell"],one["cell"])]+=1
        for arm in ("silent","remind"):
            a=normal[arm]
            b=one[arm]
            require(a["initial_remaining_calls"]>b["initial_remaining_calls"]==1 and
                    b["actual_new_calls"]<=1 and a["actual_new_calls"]<=a["initial_remaining_calls"],"non-distinct cap")
            if arm=="silent":
                hist[a["initial_remaining_calls"]]+=1
            bind[arm+"_normal_used_more_than_one"]+=int(a["actual_new_calls"]>1)
            bind[arm+"_restricted_stopped_by_call_budget"]+=int(b["termination"]=="actor_budget")
            change[arm+"_outcome_changed"]+=int(a["y"]!=b["y"])
            # Greedy coupling must preserve every action actually executed under the restricted cap.
            aa=read(root/"pairs"/pair["episode_id"]/"normal"/(arm+".json"))
            bb=read(root/"pairs"/pair["episode_id"]/"one"/(arm+".json"))
            def trace(raw):
                return [(e["proposed"],e["executed"],e["observation"],e["state"]) for e in raw["events"]]
            require(trace(aa)[:len(bb["events"])]==trace(bb),"coupled prefix drift outside intended budget cutoff")
    interval=None
    if differences:
        rng=random.Random(8100091)
        res=sorted(sum(rng.choices(differences,k=len(differences)))/len(differences) for _ in range(5000))
        interval=[res[124],res[4874]]
    report={"n_comparable":len(differences),"normal_remaining_after_draft":hist,
            "actual_call_utilization":bind,"success_changes":change,
            "cell_transitions":{k+" -> "+v:n for (k,v),n in pair_cells.items()},
            "mean_net_utility_interaction":primary["mean_change_in_D"],
            "descriptive_task_bootstrap_95":interval,
            "bootstrap_note":"Conditional first-hit checkpoint population; 5000 paired task resamples. Timing noise is included.",
            "all_shorter_action_prefixes_agree":True}
    save(root/"budget-contrast.json",report)
    print(report)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--root",type=Path,required=True)
    summarize(cli.parse_args().root)

