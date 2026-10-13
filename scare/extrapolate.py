import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .io import write_json
from .trajectory import predict_delta, project_delta, endpoint_delta


def index(path):
    result = {}
    for file in sorted(Path(path).glob("*.safetensors")):
        with safe_open(file, framework="pt") as source:
            for key in source.keys():
                if key in result:
                    raise ValueError("Duplicate parameter key")
                result[key] = file
    if not result:
        raise ValueError("No safetensors weights found")
    return result


def tensor(mapping, key):
    with safe_open(mapping[key], framework="pt") as source:
        return source.get_tensor(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--mode", choices=["rank1", "projection", "endpoint"], default="rank1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    steps = np.asarray(args.steps)
    if len(steps) != len(args.checkpoints) or len(steps) < 3 or steps[0] <= 0 or np.any(np.diff(steps) <= 0):
        parser.error("Provide at least three checkpoints in strictly increasing positive step order")
    if args.target <= steps[-1]:
        parser.error("The target must exceed the last observed step")
    base = Path(args.base)
    bi = index(base)
    ci = [index(path) for path in args.checkpoints]
    if any(set(mapping) != set(bi) for mapping in ci):
        raise ValueError("Checkpoint parameter keys differ from base")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model", "chat_template.jinja"):
        file = base / name
        if file.is_file():
            if file.suffix == ".json":
                value = json.loads(file.read_text())
                if isinstance(value, dict):
                    for key in ("_name_or_path", "name_or_path"):
                        value.pop(key, None)
                write_json(out / name, value)
            else:
                shutil.copyfile(file, out / name)
    buffer, mapping, stats = {}, {}, {}
    size = total_size = shard = 0

    def flush():
        nonlocal buffer, size, shard
        if not buffer:
            return
        shard += 1
        name = f"model-{shard:05d}.safetensors"
        save_file(buffer, out / name, metadata={"format": "pt"})
        mapping.update({key: name for key in buffer})
        buffer, size = {}, 0

    for key in bi:
        base_tensor = tensor(bi, key).float()
        flat = base_tensor.flatten()
        deltas = []
        for checkpoint in ci:
            weight = tensor(checkpoint, key).float()
            if weight.shape != base_tensor.shape:
                raise ValueError("Checkpoint parameter shape mismatch")
            deltas.append((weight.flatten() - flat).numpy())
        matrix = np.stack(deltas)
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite checkpoint delta")
        if args.mode == "rank1":
            delta, stat = predict_delta(matrix, steps, args.target)
        elif args.mode == "projection":
            delta, stat = project_delta(matrix), {}
        else:
            delta, stat = endpoint_delta(matrix, steps, args.target), {}
        weight = (flat + torch.from_numpy(delta)).reshape(base_tensor.shape).contiguous()
        if not torch.isfinite(weight).all():
            raise ValueError("Nonfinite reconstructed parameter")
        buffer[key] = weight
        stats[key] = stat
        size += weight.numel() * 4
        total_size += weight.numel() * 4
        if size >= 400_000_000:
            flush()
    flush()
    write_json(out / "model.safetensors.index.json", {"metadata": {"total_size": total_size}, "weight_map": mapping})
    write_json(out / "extrapolation.json", {"steps": args.steps, "target": args.target, "mode": args.mode, "per_tensor": stats})


if __name__ == "__main__":
    main()
