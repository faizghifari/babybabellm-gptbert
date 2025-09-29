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

## End-to-end example: train → convert (English)

1. **Train an English checkpoint** using DDP. This example assumes two GPUs on a single host and saves checkpoints under `model_checkpoints/`:

	```bash
	CONFIG=configs/base.json \
	PREPROCESS_DATASET_TYPE=monolingual PREPROCESS_MONO_LANG=eng \
	N_GPUS=2 NAME="eng-ddp-baseline" MAX_STEPS=1250 \
	./scripts/run_train_multigpu.sh
	```

	Monitor the run in the console or W&B (unless disabled). Checkpoints will land in `model_checkpoints/mono_eng_small_1_2*.bin` when the job finishes.

2. **Convert the trained weights to Hugging Face format** and optionally push them to the Hub. The command below keeps both main & EMA variants, copies the raw checkpoints, and injects the causal attention mask into the remote code for safer generation:

	```bash
	HF_USERNAME=your-hub-handle \
	python scripts/convert_and_push_mono_1_2.py \
	  --languages eng \
	  --variant both \
	  --include-raw \
	  --force-causal-mask \
	  --repo-template "{username}/babybabellm-gptbert-{lang}{causal_suffix}" \
	  --default-variant ema \
	  --push
	```

	- Drop `--push` to stage files locally under `converted/` for inspection before uploading.
	- Remove `--force-causal-mask` if you want to preserve the original bidirectional attention behavior (not recommended for causal evaluation).

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

## Checkpoint conversion

- Use `scripts/convert_and_push_mono_1_2.py` to package trained checkpoints for the Hugging Face Hub or local export.
- The script autodetects mono checkpoints in `model_checkpoints/`, saves the original tokenizer + remote code, and can optionally push directly to the Hub (`--push`).
- Provide a Hub username via `--username` or set `HF_USERNAME` in your environment before running the script.
- For quick smoke tests you can run it locally: `python scripts/convert_and_push_mono_1_2.py --variant both --languages eng --include-raw`.
- Add `--rehost-prefix <org/repo-prefix>` to standardize remote code on existing Hub repos without touching local checkpoints.

### Example

```bash
HF_USERNAME=your-hub-handle \
python scripts/convert_and_push_mono_1_2.py \
	--variant both \
	--languages eng deu \
	--include-raw \
	--checkpoint-dir model_checkpoints \
	--repo-template "{username}/babybabellm-gptbert-{lang}{causal_suffix}" \
	--default-variant ema \
	--push
```

Key arguments:

- `--username` / `HF_USERNAME`: Hub owner for the target repos (required when pushing).
- `--checkpoint-dir`, `--checkpoint-glob`, `--checkpoint-regex`: where and how to discover training checkpoints.
- `--languages`: optional filter to restrict conversion to specific language codes.
- `--variant`: choose `main`, `ema`, or `both` to control which checkpoints are exported.
- `--default-variant`: determines which weights populate `model.safetensors` and legacy `.bin`.
- `--tokenizer-id` / `--tokenizer-path`: override tokenizer discovery with a Hub repo or local file.
- `--repo-template`: customize naming; uses `{username}`, `{lang}`, `{variant}`, `{variant_suffix}`, `{causal_suffix}` placeholders.
- `--push`: upload to the Hub; omit to stage files under `converted/` locally.
- `--force-causal-mask`: insert a triangular future mask in the remote code so the exported model never attends to future tokens (mirrors the debugger’s forced causal mode).
- `--causal`: append a causal wrapper and suffix repos with `-causal` when used with the default template.
- `--rehost-prefix`: rebuild remote code for existing Hub repos instead of converting local weights.

## Troubleshooting

- If you change `vocab_size` in the config, clear old shards or run preprocessing again.
- Port conflicts: set `MASTER_PORT`.
