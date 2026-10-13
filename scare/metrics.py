import re
from collections import defaultdict


def repetition(text):
    tokens = re.findall(r"\w+", text.lower())
    grams = [tuple(tokens[i:i + 10]) for i in range(max(0, len(tokens) - 9))]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["language"]].append(record)
    metrics = {}
    for language, rows in sorted(groups.items()):
        n = len(rows)
        values = {
            "n": n,
            "accuracy": sum(row["correct"] for row in rows) / n,
            "truncation_rate": sum(row["truncated"] for row in rows) / n,
            "mean_generated_tokens": sum(row["new_tokens"] for row in rows) / n,
            "duplicate_10gram_rate": sum(repetition(row["prediction"]) for row in rows) / n,
        }
        levels = {}
        for level in ("low", "medium", "high", "top"):
            selected = [row for row in rows if row.get("difficulty_bucket") == level]
            if selected:
                levels[level] = sum(row["correct"] for row in selected) / len(selected)
        if levels:
            values["levels"] = levels
        if len(levels) == 4:
            values["difficulty_weighted_accuracy"] = sum(levels[level] * weight for level, weight in zip(("low", "medium", "high", "top"), (1, 2, 4, 8))) / 15
        metrics[language] = values
    return metrics
