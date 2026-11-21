"""Live, resource-bounded continuations for the mini task."""

import random
import time
from copy import deepcopy

from .models import FrozenQwen, xml_parser
from .records import read, verify
from .replay import advance, nodes, parse, require, restore


def load_actor(manifest):
    info = manifest["actor"]
    actor = FrozenQwen(info["actor"], info["max_new_tokens"], info["temperature"])
    require(actor.info == info, "current actor/config/device differs from frozen collection manifest")
    return actor


def branch(run, manifest, item, turn, actor, arm, reminder, seed, total_tokens, total_tools):
    pre, draft, generation = restore(run, item, turn)
    state = deepcopy(pre["state"])
    messages = deepcopy(pre["messages"])
    parser = xml_parser()
    max_steps = manifest["max_steps"]
    max_calls = max_steps
    used_calls = turn - int(manifest.get("cohort") == "controlled_fixed_prefix")
    used_in, used_out = draft["used_input_tokens"], draft["used_output_tokens"]
    if arm == "free":
        used_calls -= 1
        used_in, used_out = pre["used_input_tokens"], pre["used_output_tokens"]
    tools = pre["used_tool_calls"]
    prefix = {
        "input_tokens": used_in,
        "output_tokens": used_out,
        "actor_calls": used_calls,
        "tool_calls": tools,
    }
    base_seconds = sum(
        read(folder / "generation.json")["seconds"]
        for i, folder, _ in nodes(run, item)
        if i < turn or (i == turn and arm != "free")
    )
    ledger, events = [], []
    revision_calls = 0
    inference_seconds = tool_seconds = 0.0
    end = "budget"
    require(
        used_in + used_out <= total_tokens and tools <= total_tools,
        "prefix already exceeds study budget",
    )
    start = time.monotonic()
    for i in range(turn, max_steps + 1):
        if i == turn and arm == "silent":
            gen, action, error = generation, draft["a0"], draft["error"]
        else:
            current = deepcopy(messages)
            revision = i == turn and arm != "free"
            stream = "revision" if revision else f"suffix:{i - turn}"
            if revision:
                current[-1]["content"] += "\n\nAdvisory note for this proposal only: " + reminder
            prompt_tokens = actor.count_prompt_tokens(current)
            reserve = prompt_tokens + manifest["actor"]["max_new_tokens"]
            if used_calls >= max_calls or used_in + used_out + reserve > total_tokens:
                end = "actor_budget"
                break
            decode_seed = random.Random(f"{seed}:{stream}").randrange(2**31)
            used_calls += 1
            revision_calls += int(revision)
            request = {"stage": stream, "seed": decode_seed, "messages": current}
            ledger.append(request)
            try:
                gen = actor.generate(current, decode_seed)
            except Exception as exc:
                request.update(status="error", error_type=type(exc).__name__)
                end = "actor_error"
                break
            require(
                gen["prompt_tokens"] == prompt_tokens
                and gen["completion_tokens"] <= manifest["actor"]["max_new_tokens"],
                "actor token accounting mismatch",
            )
            request.update(generation=gen, status="ok")
            used_in += gen["prompt_tokens"]
            used_out += gen["completion_tokens"]
            inference_seconds += gen["seconds"]
            action, error = parse(gen["text"], parser)
        # Continue using the executed answer and the original user message. The
        # temporary advisory note and the discarded draft never enter history.
        messages.append({"role": "assistant", "content": gen["text"]})
        if tools + len(action["commands"]) > total_tools:
            end = "tool_budget"
            break
        tool_start = time.monotonic()
        obs, executed, reason = advance(state, action, error)
        tool_seconds += time.monotonic() - tool_start
        tools += len(executed["commands"]) if executed else 0
        events.append(
            {
                "turn": i,
                "proposed": action,
                "executed": executed,
                "parse_error": error,
                "observation": obs,
                "state": deepcopy(state),
                "remaining_calls": max_calls - used_calls,
            }
        )
        if reason:
            end = reason
            break
        messages.append({"role": "user", "content": "\n".join(obs)})
    score = (
        {"status": "unavailable", "reward": None, "reason": "actor_error"}
        if end == "actor_error"
        else verify(item["task"], state)
    )
    return {
        "arm": arm,
        "seed": seed,
        "a0_digest": draft["a0_digest"],
        "termination": end,
        "score": score,
        "state": state,
        "events": events,
        "requests": ledger,
        "final_messages": messages,
        "revision_calls": revision_calls,
        "common_prefix": prefix,
        "usage": {
            "input_tokens": used_in,
            "output_tokens": used_out,
            "tool_calls": tools,
            "actor_calls": used_calls,
            "common_actor_seconds": base_seconds,
            "new_actor_seconds": inference_seconds,
            "new_tool_seconds": tool_seconds,
            "branch_wall_seconds": time.monotonic() - start,
            "cost_usd": None,
        },
        "limits": {
            "actor_calls": max_calls,
            "turns": max_steps,
            "total_tokens": total_tokens,
            "tool_calls": total_tools,
        },
    }


def revision_fits(manifest, item, turn, actor, pre, draft, reminder, total_tokens):
    calls = turn - int(manifest.get("cohort") == "controlled_fixed_prefix")
    messages = deepcopy(pre["messages"])
    messages[-1]["content"] += "\n\nAdvisory note for this proposal only: " + reminder
    return (
        calls < manifest["max_steps"]
        and draft["used_input_tokens"]
        + draft["used_output_tokens"]
        + actor.count_prompt_tokens(messages)
        + manifest["actor"]["max_new_tokens"]
        <= total_tokens
    )
