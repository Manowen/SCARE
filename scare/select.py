import argparse
import math
from .io import read_rows, write_json


def select_candidate(candidates):
    if not candidates:
        raise ValueError("No development candidates")
    if any(not math.isfinite(row["accuracy"]) or not 0 <= row["accuracy"] <= 1 or row["target"] <= 0 for row in candidates):
        raise ValueError("Accuracy must be a finite fraction and targets must be positive")
    if len({row["target"] for row in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate targets")
    return max(candidates, key=lambda row: (row["accuracy"], -row["target"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    selected = select_candidate(read_rows(args.candidates))
    write_json(args.output, {"target": selected["target"], "accuracy": selected["accuracy"]})


if __name__ == "__main__":
    main()
