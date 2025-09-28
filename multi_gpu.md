# Multi-GPU Training (DDP)

Use `scripts/run_train_multigpu.sh` to launch training across multiple GPUs. It will build the tokenizer and shards if missing, then start DDP training.

## Configs

- Tier‑1 monolingual: `configs/base.json`
- Tier‑2 & Tier‑3 monolingual: `configs/small.json`
- Multilingual: `configs/multilingual.json`

The launcher reads `vocab_size` from your config to keep tokenizer/shards and model in sync.

## Quick start (single machine)

```bash
# Indonesian, 2 GPUs, short smoke test
CONFIG=configs/base.json \
PREPROCESS_DATASET_TYPE=monolingual PREPROCESS_MONO_LANG=ind \
N_GPUS=2 NAME="ind-2gpu-100steps" MAX_STEPS=100 \
./scripts/run_train_multigpu.sh
```

Defaults:
- Multi‑GPU only (exits if `N_GPUS <= 1`).
- W&B run name = `NAME` (override with `WANDB_NAME`; disable with `WANDB_DISABLED=1`).
- Balanced hybrid by default: numerator = denominator/2 (denominator defaults to `N_GPUS`).
- Training length: aim ~10 epochs; if unsure, set `MAX_STEPS=1250`.

## SLURM

- Monolingual: `sbatch scripts/run_multigpu_mono.slurm <lang>`
- Multilingual (small): `sbatch scripts/run_multigpu_multismall.slurm`
- Multilingual (all): `sbatch scripts/run_multigpu_multiall.slurm`

You can pass overrides via `--export`, e.g., `NAME`, `MAX_STEPS`, `N_GPUS`.

## Common env vars

- CONFIG: model config path
- NAME: run/W&B name
- N_GPUS: GPUs to launch
- PREPROCESS_DATASET_TYPE: monolingual | multilingual_small | multilingual_all
- PREPROCESS_MONO_LANG: required for monolingual

## Troubleshooting

- If you change `vocab_size` in the config, clear old shards or run preprocessing again.
- Port conflicts: set `MASTER_PORT`.
