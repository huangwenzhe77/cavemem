"""Learn a legal-command shortlist from successful training trajectories."""

import json
import math
import re
from time import monotonic
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .records import digest, read, save


def words(text):
    return re.findall(r"[a-z0-9]+",text.lower())


def shape(text):
    return re.sub(r"\b[a-z]+ \d+\b","@",text)


def entities(text):
    return re.findall(r"\b([a-z]+) (\d+)\b",text.lower())


def feature_matrix(goal, observation, history, commands, shapes, vocabulary=(), context_vocabulary=()):
    objective = words(goal)
    tokens = set(objective)
    seen = set(entities(observation))
    visited = Counter(history)
    prev = entities(history[-1]) if history else []
    labels = [shape(a) for a in history]
    quantities = Counter(labels)
    prior = [Counter(shape(a) for a in history if token in words(a)) for token in objective]
    codes = {s:i for i,s in enumerate(shapes)}
    size = len(shapes)
    x = np.zeros((len(commands), (11 if vocabulary else 3)*size+31+2*len(context_vocabulary)),dtype=np.float32)
    extra = []
    if vocabulary:
        goal_entities = list(dict.fromkeys(w for w in objective if w in vocabulary))
        role = [set(goal_entities[:1]),set(goal_entities[1:])]
        for source in (commands,history):
            by = [Counter(shape(a) for a in source if r.intersection(words(a))) for r in role]
            by.append(Counter(shape(a) for a in source if not tokens.intersection(n for n,_ in entities(a))))
            extra += [math.log1p(count[label]) for count in by for label in shapes]
        by = [Counter(shape(a) for a in history[-5:] if r.intersection(words(a))) for r in role]
        extra += [math.log1p(count[label]) for count in by for label in shapes]
    if context_vocabulary:
        visible = set(words(observation))
        extra.extend(float(t in context) for context in (tokens,visible) for t in context_vocabulary)
    for ix,a in enumerate(commands):
        s = shape(a)
        if s in codes:
            x[ix,codes[s]] = 1
        if labels and labels[-1] in codes:
            x[ix,size+codes[labels[-1]]] = 1
        x[ix,2*size:3*size] = [math.log1p(quantities[s]) for s in shapes]
        v = [math.log1p(len(history)),math.log1p(len(commands)),math.log1p(visited[a]),
             float(bool(history) and a==history[-1]),math.log1p(quantities[s])]
        es = entities(a)
        for count in range(2):
            if count>=len(es):
                v += [0.]*13
                continue
            pair = es[count]
            name, number = pair
            index = objective.index(name) if name in tokens else None
            suffix = f"{name} {number}"
            v += [1.,float(name in tokens),0. if index is None else (index+1)/max(len(objective),1),
                  float(pair in seen),float(pair in prev),float(any(n==name for n,_ in seen)),
                  math.log1p(sum(v for b,v in visited.items() if suffix in b)),
                  math.log1p(sum(v for b,v in visited.items() if b.endswith(suffix))),
                  float(number=="1"),float(number=="2"),len(es)/3,
                  math.log1p(sum(quantities[g] for g in shapes if g.endswith("@"))),
                  math.log1p(sum(prior[index].values())) if index is not None else 0.]
        x[ix,3*size:] = v+extra
    return torch.from_numpy(x)


class Ranker(nn.Module):
    def __init__(self,shapes,vocabulary=(),context_vocabulary=()):
        super().__init__()
        self.shapes = shapes
        self.vocabulary = frozenset(vocabulary)
        self.context_vocabulary = context_vocabulary
        width=(11 if vocabulary else 3)*len(shapes)+31+2*len(context_vocabulary)
        self.net = nn.Sequential(nn.Linear(width,96),nn.ReLU(),nn.Linear(96,48),nn.ReLU(),
                                 nn.Linear(48,1))

    def forward(self,x):
        return self.net(x).squeeze(-1)

    def features(self,goal,observation,history,menu):
        return feature_matrix(goal,observation,history,menu,self.shapes,self.vocabulary,self.context_vocabulary)

    def ordering(self,ep):
        actions = ep.commands
        history = [r["action"] for r in ep.history]
        with torch.inference_mode():
            scores = self(self.features(ep.goal,ep.observation,history,actions)).tolist()
        return sorted(zip(actions,scores),key=lambda p:(-p[1],p[0]))


def train(root, seed, rich=False, lexical=False, recovery=False):
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    files = sorted((root/"demos").glob("*.json"))
    traces = [r for p in files if (r:=read(p))["success"]]
    if len(traces)<len(files)/2 or len(traces)<12:
        raise ValueError("insufficient successful expert demonstrations")
    shapes = sorted({shape(a) for t in traces for r in t["rows"] for a in r["commands"]})
    vocab = sorted({n for t in traces for r in t["rows"] for a in r["commands"] for n,_ in entities(a)}) if rich else []
    contexts = sorted({w for r in traces for w in words(r["goal"])}) if lexical else []
    ranker = Ranker(shapes,vocab,contexts)
    entries = []
    correction = [read(p) for p in sorted((root/"recovery").glob("*.json"))] if recovery else []
    for row in traces+correction:
        history = []
        observed = set()
        for r in row["rows"]:
            key = (r["observation"],tuple(r["commands"]),r["action"],tuple(history[-5:]))
            if key not in observed:
                x = ranker.features(row["goal"],r["observation"],history,r["commands"])
                entries.append((x,r["commands"].index(r["action"])))
                observed.add(key)
            history.append(r.get("executed",r["action"]))
    optimizer = torch.optim.AdamW(ranker.parameters(),lr=.001)
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(28 if lexical else 24 if rich else 14):
        loss_sum, correct = 0., 0
        indexes = torch.randperm(len(entries),generator=gen).tolist()
        for j in range(0,len(indexes),48):
            batch = [entries[i] for i in indexes[j:j+48]]
            lengths = [len(x) for x,_ in batch]
            logits = ranker(torch.cat([x for x,_ in batch]))
            scores = nn.utils.rnn.pad_sequence(logits.split(lengths),batch_first=True,padding_value=-1e9)
            targets = torch.tensor([y for _,y in batch])
            loss = nn.functional.cross_entropy(scores,targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(batch)
            correct += (scores.argmax(-1)==targets).sum().item()
        print("RANKER",epoch+1,len(entries),round(loss_sum/len(entries),5),round(correct/len(entries),5),flush=True)
    torch.save(ranker.state_dict(),root/"ranker.pt")
    save(root/"ranker.json",{"shapes":shapes,"vocabulary":vocab,"context_vocabulary":contexts,
                             "games":len(traces),"recovery_games":len(correction),"decisions":len(entries),
                             "training_only":True,"seed":seed})


def load(root):
    torch.set_num_threads(2)
    package = read(root/"ranker.json")
    model = Ranker(package["shapes"], package.get("vocabulary", ()),package.get("context_vocabulary", ()))
    model.load_state_dict(torch.load(root/"ranker.pt",map_location="cpu",weights_only=True))
    return model.eval()


class RankedActor:
    def __init__(self, actor, ranker, top, repeat_aware=False, examples=(), indexed=False):
        self.actor, self.ranker, self.top = actor, ranker, top
        self.repeat_aware = repeat_aware
        self.examples = examples
        self.demo_cache = {}
        self.indexed = indexed

    def demo(self,goal):
        if goal not in self.demo_cache:
            query = set(words(goal))-self.ranker.vocabulary
            nearest = max(self.examples,key=lambda d:len(query & d[0])/max(1,len(query | d[0])))
            self.demo_cache[goal] = nearest[1]
        return self.demo_cache[goal]

    def propose(self, ep, reminder=None, original=None, **sampling):
        start = monotonic()
        ranked = self.ranker.ordering(ep)
        if self.repeat_aware:
            menu = digest(ep.commands)
            counts = Counter(r["action"] for r in ep.history if r["menu"]==menu)
            # The unfiltered menu is the recurrence key; the learned shortlist changes over time.
            low = min(counts[a] for a,_ in ranked)
            ranked = [(a,s) for a,s in ranked if counts[a]==low]
        choices = [a for a,_ in ranked[:self.top]]
        if original and original not in choices:
            choices.append(original)
        obs = (ep.observation + "\nCandidate commands ranked by a model trained on earlier successful games: "
               + ", ".join(choices) + ". Use the task, current observation and history to choose.")
        if self.repeat_aware:
            obs += " Consider the first candidate unless the task or current evidence favors another."
        if self.examples:
            obs += "\nA successful training-game example for planning only: " + self.demo(ep.goal)
        if callable(reminder):
            reminder = reminder(ep, choices)
        if self.indexed:
            return self.select_index(ep, obs, choices, reminder, original, start)
        view = SimpleNamespace(goal=ep.goal,history=ep.history,observation=obs,commands=choices)
        return {**self.actor.propose(view,reminder,original,**sampling),
                "ranked_candidates":choices}

    def select_index(self,ep,obs,choices,reminder,original,start):
        tokenizer = self.actor.tokenizer
        model = self.actor.model
        history = "\n".join(f"{r['action']} → {r['observation']}" for r in ep.history[-8:])
        actions = "\n".join(f"{i+1}. {a}" for i,a in enumerate(choices))
        user = (f"Task: {ep.goal}\nRecent transitions:\n{history}\nCurrent observation: {obs}"
                f"\nCandidate actions:\n{actions}")
        if reminder is not None:
            if original is not None:
                user += f"\nOriginal proposal: {original}"
            user += f"\nAdvisory memory: {reminder}"
            marker = "The memory matcher found this currently legal candidate: "
            if marker in reminder:
                candidate = reminder.rsplit(marker,1)[1].split(". ",1)[0]
                if candidate in choices:
                    user += f"\nAdvisory candidate number: {choices.index(candidate)+1}."
        user += "\nChoose the single best next action. Reply with just its number."
        msgs = [{"role":"system","content":"You are a household task agent. "
                 "Use the task and visible environment to select one numbered action."},
                {"role":"user","content":user}]
        prompt = tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        enc = tokenizer(prompt,return_tensors="pt",add_special_tokens=False).to(model.device)
        options = [tokenizer.encode(str(i+1),add_special_tokens=False) for i in range(len(choices))]
        if not all(len(z)==1 for z in options):
            raise ValueError("candidate labels must tokenize to one token")
        with torch.inference_mode():
            logits = model(**enc,use_cache=False,logits_to_keep=1).logits[0,-1]
            index = logits[[z[0] for z in options]].argmax().item()
        return {"action":choices[index],"allowed":choices,"ranked_candidates":choices,
                "prompt_digest":digest(prompt),"input_tokens":enc.input_ids.shape[1],"output_tokens":1,
                "choice_logits":logits[[z[0] for z in options]].float().tolist(),
                "seconds":monotonic()-start}


def examples(root, ranker):
    demos = []
    for p in sorted((root/"demos").glob("*.json")):
        r = read(p)
        if not r["success"]:
            continue
        types = set(words(r["goal"]))-ranker.vocabulary
        path = " → ".join(shape(a["action"]) for a in r["rows"])
        demos.append((types, r["goal"]+" | "+path))
    return demos
