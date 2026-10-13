# ScaRE

Core implementation of **Scaffolded Rank-1 Extrapolation** for multilingual reasoning.

Train briefly with English solution prefixes, construct models from early checkpoint trajectories, and select the extrapolation target using hint-free development accuracy. Evaluate the selected model without hints.

## Contents

- `scare/prompts.py`: plain-text prompts and fixed/decaying hint schedules.
- `scare/train.py`, `scare/trainer.py`: GRPO training and numerical safeguards.
- `scare/trajectory.py`, `scare/extrapolate.py`: temporal rank-1 extrapolation, projection, and endpoint scaling.
- `scare/grading.py`, `scare/evaluate.py`, `scare/metrics.py`: answer checking, greedy evaluation, accuracy, weighted accuracy, truncation, and repetition.
- `scare/select.py`: development-set selection with smaller-target tie breaking.
- `tests/`: synthetic checks of core operations.

Datasets, weights, predictions, logs, and cluster-specific infrastructure are not included. Supply locations as command-line arguments. Training checkpoints contain model weights only; optimizer-state resumption is not supported.

## Environment

Use Python 3.12 and a GPU-compatible PyTorch build matching `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Dependencies are pinned to the implementation environment. Core tests run on CPU. Training requires eight GPU processes and sufficient memory for FP32 parameters, optimizer states, and generation. Training uses BF16 computation, FP64 distributed gradient summation, and disables cuDNN SDPA. It does not use vLLM.

## Input format

Provide UTF-8 JSONL files with these fields:

| Field | Meaning |
| --- | --- |
| `id` | Unique question identifier within a language |
| `language` | Question-language code |
| `question` | Question text |
| `answer` | Reference mathematical answer |
| `hint_english` | English solution prefix for hinted training |
| `hint_prefixes` | Increasing paragraph-complete prefixes ending at `hint_english`, required for gradual withdrawal |
| `difficulty_bucket` | Optional evaluation level: `low`, `medium`, `high`, or `top` |

Construct hints from English reference solutions at complete paragraph boundaries. The experimental protocol uses 40–512 words and rejects prefixes containing explicit final answers. These scripts consume prepared, aligned rows; translation, alignment, filtering, and benchmark-overlap checks must be completed beforehand. Keep development and test problems disjoint from training.

The fixed schedule retains the complete prefix. Decay applies a cosine word budget from steps 1–200, reaching zero at step 200, while retaining complete prefix boundaries. Evaluation always omits hints.

## Training

Run from this directory, setting `MODEL` and `TRAIN_DATA` to your inputs:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m torch.distributed.run --standalone --nproc_per_node=8 --module scare.train \
  --model "$MODEL" --data "$TRAIN_DATA" --language ko \
  --schedule fixed --steps 50 --seed 42 --output outputs/fixed
```

Each step uses 20 questions with four responses each. The learning rate is `1e-6`; rewards measure answer correctness, without KL or language-consistency penalties. Checkpoints are saved at steps 10, 25, 50, and the final step. For full-training controls, use `--steps 250` and `--schedule none`, `fixed`, or `decay`, with separate output directories.

## Model construction

Use the same base model as training, in a local safetensors directory:

```bash
python -m scare.extrapolate --base "$BASE_MODEL_DIRECTORY" \
  --checkpoints outputs/fixed/checkpoint-10 outputs/fixed/checkpoint-25 outputs/fixed/checkpoint-50 \
  --steps 10 25 50 --target 250 --mode rank1 --output outputs/rank1-250
```

Repeat with target 100 for the other candidate. `projection` reconstructs the last observed rank-1 coefficient without extrapolation. `endpoint` scales the final update by `target / last_step`. Rank one refers to the temporal matrix of flattened updates, not individual weight matrices. The leading singular direction is computed through the temporal Gram matrix.

Construction processes one tensor at a time and writes sharded FP32 weights. Memory depends on the largest tensor and checkpoint count. Existing output directories are not overwritten.

## Evaluation and selection

```bash
python -m scare.evaluate --model outputs/rank1-250 \
  --data "$DEV_DATA" --output outputs/dev-250
```

Evaluate candidates on the same hint-free development split in the training language. Prepare candidate JSONL objects with `target` and `accuracy` (a fraction from zero to one):

```bash
python -m scare.select --candidates "$DEV_CANDIDATES" --output outputs/selection.json
```

Highest development accuracy wins, breaking ties toward the smaller target. Do not select on test accuracy. Evaluate the selected model on test data using `scare.evaluate`. The matched-target endpoint control uses the target chosen for its corresponding rank-1 trajectory.

Evaluation uses one GPU, greedy decoding, and 4,096 generated tokens. Independent evaluations can use separate devices. Rates are fractions. Difficulty-weighted accuracy requires all four levels and uses weights 1, 2, 4, and 8. Repetition is the duplicate fraction of lowercased word 10-grams.

## Compute accounting

Training records timed training GPU-hours. Evaluation records generation and scoring after loading. Additionally measure construction and all development evaluations, including unselected candidates, for method-level comparisons. Exclude final tests consistently. Report CPU-only construction separately. Local timers do not replace end-to-end allocation accounting.
