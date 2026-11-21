"""TextWorld backend for CAVE using the official ALFWorld game files."""

import hashlib
import os
import re
import time
from collections import Counter
from pathlib import Path

from .models import FrozenQwen
from .records import digest


MAX_STEPS = 50
SYSTEM = (
    "You act in an ALFWorld household. Complete the task by issuing one exact command "
    "from the available list. Use visible feedback and previous actions to track progress. "
    "Get the requested object, perform any requested treatment, then finish the task. "
    "Avoid repeating actions that gave no progress. An optional advisory memory is a "
    "suggestion from another game. Use it only if it fits this game's visible situation."
)


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Episode:
    def __init__(self, data, config, spec):
        import yaml
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

        self.data, self.config, self.spec = data, config, spec
        game = (data / spec["game"]).resolve()
        game.relative_to(data.resolve())
        if sha_file(game) != spec["sha256"]:
            raise ValueError("ALFWorld game differs from the frozen cohort")
        cfg = yaml.safe_load(config.read_text())
        cfg["dataset"]["data_path"] = str(game.parent)
        cfg["env"]["domain_randomization"] = False
        cfg["general"]["training_method"] = "dqn"
        cfg["rl"]["training"]["max_nb_steps_per_episode"] = MAX_STEPS
        os.environ["ALFWORLD_DATA"] = str(data)
        games = AlfredTWEnv(cfg)
        if games.game_files != [str(game)]:
            raise ValueError("unexpected game selection")
        self.env = games.init_env(batch_size=1)
        self.env.request_infos.facts = True
        self.env.request_infos.moves = True
        self.env.seed(spec["seed"])
        obs, self.info = self.env.reset()
        self.initial = self.observation = obs[0]
        self.goal = self.initial.partition("Your task is to:")[2].strip()
        if not self.goal:
            raise ValueError("missing public goal")
        self.history = []
        self.done = False
        self.initial_signature = self.signature()

    @property
    def commands(self):
        return sorted(self.info["admissible_commands"][0])

    @property
    def won(self):
        return bool(self.info["won"][0])

    @property
    def terminal(self):
        return self.done or self.won or len(self.history) >= MAX_STEPS

    def signature(self):
        facts = self.info["facts"][0]
        moves = self.info["moves"][0]
        if not facts or moves is None:
            raise ValueError("missing environment state evidence")
        return digest([sorted(map(str, facts)), moves, self.observation, self.commands,
                       self.done, self.won])

    def advance(self, action):
        if action not in self.commands or self.terminal:
            raise ValueError(f"invalid ALFWorld execution: {action}")
        start = time.monotonic()
        menu = digest(self.commands)
        obs, _, done, self.info = self.env.step([action])
        self.observation, self.done = obs[0], bool(done[0])
        row = {"action": action, "observation": self.observation, "menu": menu,
               "state": self.signature(), "seconds": time.monotonic() - start}
        self.history.append(row)
        return row

    def restore(self, prefix, signature):
        other = Episode(self.data, self.config, self.spec)
        try:
            if other.initial_signature != self.initial_signature:
                raise ValueError("initial state replay mismatch")
            for row in prefix:
                if other.advance(row["action"])["state"] != row["state"]:
                    raise ValueError("intermediate state replay mismatch")
            if other.signature() != signature:
                raise ValueError("checkpoint replay mismatch")
            return other
        except Exception:
            other.env.close()
            raise


class Actor(FrozenQwen):
    def __init__(self, path):
        super().__init__(path, max_new_tokens=24, temperature=0)
        self.info.update(system_prompt=SYSTEM, template="alfworld-legal-v1",
                         template_digest=digest(SYSTEM), stop_strings=["eos"])

    def propose(self, ep, reminder=None, original=None, *, seed=None, temperature=None):
        import torch

        start = time.monotonic()
        commands = ep.commands
        menu = digest(commands)
        repeats = Counter(r["action"] for r in ep.history if r["menu"] == menu)
        minimum = min(repeats[c] for c in commands)
        allowed = [c for c in commands if repeats[c] == minimum]
        history = "\n".join(f"{r['action']} → {r['observation']}" for r in ep.history[-12:])
        tried = ", ".join(dict.fromkeys(r["action"] for r in ep.history
                                       if r["action"].startswith(("go to ", "open "))))
        content = (f"Task: {ep.goal}\nLocations/actions tried: {tried or 'none'}\n"
                   f"Recent history:\n{history or 'none'}\nCurrent observation: {ep.observation}")
        if reminder is not None:
            content += f"\nOriginal proposal: {original}\nAdvisory memory: {reminder}"
        content += "\nAvailable commands:\n" + "\n".join(allowed)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        size = inputs.input_ids.shape[1]
        eos = self.tokenizer.eos_token_id
        trie = {}
        for action in allowed:
            pieces = self.tokenizer.encode(action, add_special_tokens=False)
            if len(pieces) >= self.max_new_tokens:
                raise ValueError("action exceeds decoder budget")
            node = trie
            for token in pieces:
                node = node.setdefault(token, {})
            node[eos] = {}

        def available(_, row):
            node = trie
            for token in row[size:].tolist():
                node = node[token]
            return list(node)

        generation = {"do_sample": seed is not None}
        if seed is not None:
            if temperature is None or not 0 < temperature <= 2:
                raise ValueError("sampled continuation needs a positive temperature")
            generation.update(temperature=temperature, top_k=0, top_p=1.)
        cuda_device = inputs.input_ids.get_device()
        with torch.inference_mode(), torch.random.fork_rng(
            devices=[cuda_device] if cuda_device >= 0 else [], enabled=seed is not None
        ):
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
                if cuda_device >= 0:
                    torch.cuda.default_generators[cuda_device].manual_seed(seed)
            result = self.model.generate(**inputs, **generation, max_new_tokens=self.max_new_tokens,
                                         eos_token_id=eos, pad_token_id=eos,
                                         prefix_allowed_tokens_fn=available)
        completion = result[0, size:]
        action = self.tokenizer.decode(completion, skip_special_tokens=True)
        if action not in allowed:
            raise ValueError("actor emitted an action outside its allowed menu")
        sampling = {} if seed is None else {"seed": seed, "temperature": temperature}
        return {"action": action, "allowed": allowed, "prompt_digest": digest(prompt), **sampling,
                "input_tokens": size, "output_tokens": completion.numel(),
                "seconds": time.monotonic() - start}


def visible_context(ep):
    goal = ep.goal.lower()
    match = re.search(r"(?:put (?:a|an|some|two) |find two |(?:look at|examine) (?:(?:a|an|the) )?|"
                      r"(?:clean|heat|cool) some )(?:(?:clean|hot|cool|cold) )?(\w+)", goal)
    target = match[1] if match else None
    held = None
    for row in ep.history:
        taken = re.fullmatch(r"take (.+?) from .+", row["action"])
        feedback = row["observation"].lower()
        if taken and "you pick up" in feedback:
            held = taken[1].split()[0]
        if ((row["action"].startswith("put ") and "you put" in feedback)
                or (row["action"].startswith("move ") and "you move" in feedback)):
            held = None
    commands = ep.commands
    visible = target and any(c.startswith(f"take {target} ") for c in commands)
    phase = "holding_target" if held and held == target else "holding_other" if held else (
        "target_visible" if visible else "search")
    checked = sorted({r["action"][6:] for r in ep.history if r["action"].startswith("go to ")})
    return {"target": target, "phase": phase, "checked": checked,
            "goal": ep.goal, "observation": ep.observation, "commands": commands}
