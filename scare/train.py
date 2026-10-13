import argparse
import os
import time
from pathlib import Path

from .io import read_rows, write_json
from .prompts import render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--schedule", choices=["none", "fixed", "decay"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 8:
        parser.error("This training configuration requires eight processes")
    if args.microbatch <= 0 or 10 % args.microbatch:
        parser.error("Microbatch must be a positive divisor of 10")
    if args.steps < 50:
        parser.error("At least 50 steps are required for the checkpoint trajectory")
    rows = [row for row in read_rows(args.data) if row["language"] == args.language]
    lookup = {row["id"]: row for row in rows}
    if not rows or len(lookup) != len(rows):
        raise ValueError("Training data must be nonempty with unique question IDs")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.distributed as dist
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, set_seed
    from trl import GRPOConfig
    from . import trainer as training
    from .grading import grade

    rank = int(os.environ.get("RANK", "0"))
    training.world, training.rank = world, rank
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.use_deterministic_algorithms(True)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    stop = {tokenizer.eos_token_id}
    if "<|im_end|>" in tokenizer.get_vocab():
        stop.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    stop = sorted(value for value in stop if value is not None)
    model.generation_config.eos_token_id = stop
    dataset = Dataset.from_list([{"prompt": render(row, args.schedule, 1)[0], "answer": row["answer"], "problem_id": row["id"]} for row in rows])

    class DynamicTrainer(training.ScaffoldTrainer):
        def _generate_and_score_completions(self, inputs):
            step = self.state.global_step + 1
            inputs = [dict(item, prompt=render(lookup[item["problem_id"]], args.schedule, step)[0]) for item in inputs]
            return super()._generate_and_score_completions(inputs)

    class Milestones(TrainerCallback):
        def on_step_end(self, config, state, control, **kwargs):
            control.should_save = state.global_step in {10, 25, 50, args.steps}
            return control

    def reward(completions, answer, **kwargs):
        return [float(grade(completion, gold)[0]) for completion, gold in zip(completions, answer)]

    config = GRPOConfig(
        output_dir=str(out), max_steps=args.steps,
        per_device_train_batch_size=args.microbatch,
        gradient_accumulation_steps=10 // args.microbatch,
        generation_batch_size=80, num_generations=4,
        max_completion_length=args.max_tokens,
        learning_rate=1e-6, lr_scheduler_type="constant", warmup_steps=0,
        beta=0.0, loss_type="grpo", scale_rewards="group",
        temperature=0.8, top_p=0.95, top_k=50,
        generation_kwargs={"eos_token_id": stop},
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=1, save_strategy="no", save_only_model=True,
        report_to="none", seed=args.seed, data_seed=args.seed,
        use_vllm=False, remove_unused_columns=False,
    )
    trainer = DynamicTrainer(model=model, args=config, reward_funcs=reward, train_dataset=dataset, processing_class=tokenizer, callbacks=[Milestones()])
    start = time.perf_counter()
    trainer.train()
    if rank == 0:
        seconds = time.perf_counter() - start
        write_json(out / "cost.json", {"seconds": seconds, "gpu_hours": world * seconds / 3600, "gpu_count": world, "steps": trainer.state.global_step})
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
