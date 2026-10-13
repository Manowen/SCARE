import math


def prompt(row, hint=""):
    question = row["question"]
    if hint:
        question += "\n\nPartial solution (not the final answer):\n" + hint
    return question + "\nPlease reason step by step, and put your final answer within \\boxed{}.\n\nSolution:\n"


def render(row, schedule, step):
    if schedule == "none":
        return prompt(row), 0
    if schedule not in {"fixed", "decay"}:
        raise ValueError("Unknown hint schedule")
    if step < 1:
        raise ValueError("Steps are one-based")
    ratio = 1.0 if schedule == "fixed" else (0.5 * (1 + math.cos(math.pi * (step - 1) / 199)) if step < 200 else 0.0)
    parts = row.get("hint_prefixes", [row["hint_english"]])
    if not parts:
        raise ValueError("At least one hint prefix is required")
    budget = math.floor(len(parts[-1].split()) * ratio)
    hint = next((part for part in reversed(parts) if len(part.split()) <= budget), "")
    return prompt(row, hint), len(hint.split())
