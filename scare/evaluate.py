import argparse
import json
import time
from pathlib import Path

from .io import read_rows, write_json
from .prompts import prompt
from .metrics import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_tokens < 1:
        parser.error("Batch size and token limit must be positive")
    rows = read_rows(args.data)
    if not rows or len({(row["id"], row["language"]) for row in rows}) != len(rows):
        raise ValueError("Evaluation rows must have unique nonempty question-language keys")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from .grading import grade
    set_seed(args.seed)
    torch.backends.cuda.enable_cudnn_sdp(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    stop = {tokenizer.eos_token_id}
    if "<|im_end|>" in tokenizer.get_vocab():
        stop.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    stop = sorted(value for value in stop if value is not None)
    rows = sorted(rows, key=lambda row: len(tokenizer.encode(prompt(row))))
    records = []
    start = time.perf_counter()
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            inputs = tokenizer([prompt(row) for row in batch], padding=True, return_tensors="pt").to("cuda")
            prefix_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                sequences = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False, eos_token_id=stop, pad_token_id=tokenizer.pad_token_id)
            for row, sequence in zip(batch, sequences):
                ids = sequence[prefix_length:].tolist()
                end = next((i for i, token in enumerate(ids) if token in stop), None)
                actual = ids if end is None else ids[:end + 1]
                text = tokenizer.decode(actual, skip_special_tokens=True)
                correct, parsed = grade(text, row["answer"])
                record = {key: row[key] for key in ("id", "language", "answer", "difficulty_bucket") if key in row}
                record.update(prediction=text, correct=correct, parsed=parsed, new_tokens=len(actual), truncated=end is None and len(actual) >= args.max_tokens)
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
    seconds = time.perf_counter() - start
    write_json(out / "summary.json", {"metrics": summarize(records), "seconds": seconds, "gpu_hours": seconds / 3600, "gpu_count": 1})


if __name__ == "__main__":
    main()
