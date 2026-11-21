"""A resettable data task. The action interpreter has no shell or file access."""
import random


TEMPLATES = {
    "source": ("ascending", "Sort these integer records from smallest to largest."),
    "target_train": ("descending", "Return all input integers in decreasing order."),
    "target_val": ("unique", "Return each distinct integer once in increasing order."),
    "target_test": ("positive", "Keep all positive integers, sorted from low to high."),
    "controlled_transfer": ("ascending", "Arrange every integer in increasing order, retaining repeated values. Submit the entire ordered list."),
}


def task(split, seed):
    rng = random.Random(seed)
    values = [rng.randint(-9, 22) for _ in range(8)]
    values[-1] = values[0]
    family, instruction = TEMPLATES[split]
    return {"family": family, "instruction": instruction, "input": values}


def initial(spec):
    return {"input": list(spec["input"]), "working": list(spec["input"]), "submitted": None}


def execute(state, commands):
    """Execute only exact DSL operations, in order, preserving all observations."""
    observations = []
    for command in commands:
        operation = command["keystrokes"].strip()
        working = state["working"]
        if operation == "SHOW":
            observations.append(f"working = {working}")
        elif operation == "ASC":
            working.sort()
            observations.append(f"ascending: {working}")
        elif operation == "DESC":
            working.sort(reverse=True)
            observations.append(f"descending: {working}")
        elif operation == "UNIQUE":
            state["working"] = list(dict.fromkeys(working))
            observations.append(f"duplicates removed: {state['working']}")
        elif operation == "POSITIVE":
            state["working"] = [n for n in working if n > 0]
            observations.append(f"positive values: {state['working']}")
        elif operation == "SUBMIT":
            state["submitted"] = working.copy()
            observations.append("submitted working list")
        else:
            observations.append("rejected: expected one exact data operation per keystrokes tag")
    return observations
