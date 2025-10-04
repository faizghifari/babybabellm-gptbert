#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRETRAIN_DIR="$ROOT_DIR/pretraining"
TOK_DIR="$ROOT_DIR/tokenizers"
CONFIG="${CONFIG:-$ROOT_DIR/configs/base.json}"

[[ -f "$CONFIG" ]] || { echo "Missing $CONFIG"; exit 1; }

# Ensure we're running multi-GPU only. Use the single-GPU script for 1 GPU setups.
if [[ "${N_GPUS:-2}" -le 1 ]]; then
  echo "This launcher is for multi-GPU only. For single-GPU training use the single-GPU training script." >&2
  exit 2
fi

# Dataset base dir and shard paths based on dataset type (match updated_preprocess.py)
PREPROCESS_DATASET_TYPE=${PREPROCESS_DATASET_TYPE:-monolingual}
PREPROCESS_MONO_LANG=${PREPROCESS_MONO_LANG:-}

case "$PREPROCESS_DATASET_TYPE" in
  monolingual)
    if [[ -z "$PREPROCESS_MONO_LANG" ]]; then
      echo "PREPROCESS_MONO_LANG is required when PREPROCESS_DATASET_TYPE=monolingual" >&2
      exit 2
    fi
    DATA_BASE_DIR="${DATA_BASE_DIR:-$ROOT_DIR/data/MONOLINGUAL/$PREPROCESS_MONO_LANG}"
    ;;
  multilingual_small)
    DATA_BASE_DIR="${DATA_BASE_DIR:-$ROOT_DIR/data/MULTILINGUAL-SMALL}"
    ;;
  multilingual_all)
    DATA_BASE_DIR="${DATA_BASE_DIR:-$ROOT_DIR/data/MULTILINGUAL-ALL}"
    ;;
  *)
    echo "Invalid PREPROCESS_DATASET_TYPE: $PREPROCESS_DATASET_TYPE" >&2
    exit 2
    ;;
esac

TRAIN_PATH="${TRAIN_PATH:-$DATA_BASE_DIR/train}"
DEV_PATH="${DEV_PATH:-$DATA_BASE_DIR/valid}"

# Preprocessing (if needed) will be handled later once NAME and VOCAB_SIZE are known

N_GPUS=${N_GPUS:-2}
MAX_STEPS=${MAX_STEPS:-15625}
SAVE_EVERY=${SAVE_EVERY:-1000}
VALIDATE_EVERY=${VALIDATE_EVERY:-1000}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NAME=${NAME:-}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/model_checkpoints"}


LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-128}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32768}
HYBRID_DENOMINATOR=${HYBRID_DENOMINATOR:-$N_GPUS}
# Default to a balanced split: half masked, half causal
HYBRID_NUMERATOR=${HYBRID_NUMERATOR:-$(( HYBRID_DENOMINATOR / 2 ))}
# Get vocab_size from config to keep tokenizer/model in sync
CONFIG_VOCAB=$(python -c 'import json,sys;print(json.load(open(sys.argv[1])).get("vocab_size", 32768))' "$CONFIG" 2>/dev/null || echo 32768)
VOCAB_SIZE="$CONFIG_VOCAB"

# Compute default NAME if not provided: monolingual -> babylm-{lang}-{num}_{den}; multilingual -> babylm-multilingual-{all|small}-{num}_{den}
if [[ -z "${NAME}" ]]; then
  if [[ "$PREPROCESS_DATASET_TYPE" == "monolingual" ]]; then
    NAME="babylm-${PREPROCESS_MONO_LANG}-${HYBRID_NUMERATOR}_${HYBRID_DENOMINATOR}"
  else
    ml_kind="small"
    if [[ "$PREPROCESS_DATASET_TYPE" == "multilingual_all" ]]; then ml_kind="all"; fi
    NAME="babylm-multilingual-${ml_kind}-${HYBRID_NUMERATOR}_${HYBRID_DENOMINATOR}"
  fi
fi

export WANDB_DISABLED=${WANDB_DISABLED:-0}
WANDB_PROJECT=${WANDB_PROJECT:-BabyLM-GPT-BERT}
WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_PROJECT
if [[ -n "$WANDB_ENTITY" ]]; then export WANDB_ENTITY; else unset WANDB_ENTITY; fi

# Ensure W&B run name follows NAME by default (can be overridden with WANDB_NAME)
export WANDB_NAME="${WANDB_NAME:-$NAME}"

WANDB_ARGS=()

COMMON_ARGS=(
  --train_path "$TRAIN_PATH"
  --valid_path "$DEV_PATH"
  --config_file "$CONFIG"
  --name "$NAME"
  --output_dir "$OUTPUT_DIR"
)

CHECKPOINT_FILENAME=${CHECKPOINT_FILENAME:-}
RESUME=${RESUME:-0}
if [[ -z "$CHECKPOINT_FILENAME" && ("$RESUME" == "1" || "${AUTO_RESUME:-0}" == "1") ]]; then
  default_ckpt="$OUTPUT_DIR/${NAME}_state_dict.bin"
  if [[ -f "$default_ckpt" ]]; then
    CHECKPOINT_FILENAME="$default_ckpt"
  fi
fi
RESUME_ARGS=()
if [[ -n "$CHECKPOINT_FILENAME" ]]; then
  echo "[resume] Using checkpoint: $CHECKPOINT_FILENAME" >&2
  RESUME_ARGS+=(--checkpoint_filename "$CHECKPOINT_FILENAME")
fi


# If shards missing (no .bin files) then prepare tokenizer and run preprocessing
if [[ -n "${PREPROCESS_MONO_LANG:-}" ]]; then
  TOK_FILE="${TOKENIZER_PATH:-$TOK_DIR/tokenizer_${PREPROCESS_MONO_LANG}_vs${VOCAB_SIZE}.json}"
else
  TOK_FILE="${TOKENIZER_PATH:-$TOK_DIR/tokenizer_multilingual_vs${VOCAB_SIZE}.json}"
fi

has_shards() {
  local d="$1"
  [[ -d "$d" ]] || return 1
  shopt -s nullglob
  local files=("$d"/*.bin)
  shopt -u nullglob
  [[ ${#files[@]} -gt 0 ]]
}

need_preprocess=0
if ! has_shards "$TRAIN_PATH" || ! has_shards "$DEV_PATH"; then
  need_preprocess=1
fi

if [[ $need_preprocess -eq 1 ]]; then
  echo "Shards missing. Running preprocessing to create shards under $DATA_BASE_DIR..."

  mkdir -p "$TOK_DIR"

  if [[ ! -f "$TOK_FILE" ]]; then
    echo "Tokenizer not found at $TOK_FILE. Training tokenizer..."
    if [[ -n "${PREPROCESS_MONO_LANG:-}" ]]; then
      python "$ROOT_DIR/tokenizers/tokenizer.py" --vocab_size "$VOCAB_SIZE" --output "tokenizer_${PREPROCESS_MONO_LANG}_vs${VOCAB_SIZE}.json"
    else
      python "$ROOT_DIR/tokenizers/tokenizer.py" --vocab_size "$VOCAB_SIZE" --output "tokenizer_multilingual_vs${VOCAB_SIZE}.json"
    fi
    if [[ ! -f "$TOK_FILE" ]]; then
      echo "Tokenizer training failed to produce $TOK_FILE" >&2
      exit 3
    fi
  else
    echo "Using existing tokenizer: $TOK_FILE"
  fi

  echo "Running streaming encoder to create shards..."
  python "$ROOT_DIR/preprocess/updated_preprocess.py" \
    --dataset_type "$PREPROCESS_DATASET_TYPE" \
    $( [[ -n "$PREPROCESS_MONO_LANG" ]] && echo "--mono_lang $PREPROCESS_MONO_LANG" ) \
    --tokenizer "$TOK_FILE" \
    --seq_length ${SEQ_LENGTH:-128} \
    --shard_size_bytes 100000000 \
    --batch_size 1000 \
    --max_workers 8 \
    --base_dir "$ROOT_DIR/data"

  echo "Preprocessing finished. Continuing to training..."
fi

# Optional: exit after preprocessing (used by SLURM multi-node to avoid races)
if [[ "${PREPROCESS_ONLY:-0}" == "1" ]]; then
  echo "PREPROCESS_ONLY=1 set; exiting before training."
  exit 0
fi

echo "Launching $N_GPUS GPUs"
export MASTER_ADDR MASTER_PORT
TORCHRUN_EXTRA_ARGS=${TORCHRUN_EXTRA_ARGS:-}
exec torchrun --nproc_per_node="$N_GPUS" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" $TORCHRUN_EXTRA_ARGS \
  "$PRETRAIN_DIR/train_multi_gpu.py" \
  "${COMMON_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  --tokenizer_path "$TOK_FILE" \
  --hybrid_denominator "$HYBRID_DENOMINATOR" \
  --hybrid_numerator "$HYBRID_NUMERATOR" \
  --optimizer lamb \
  --optimizer_beta1 0.9 \
  --optimizer_beta2 0.98 \
  --optimizer_eps 1e-8 \
  --weight_decay 0.1 \
  --warmup_proportion 0.016 \
  --learning_rate 1e-2 \
  --seq_length 128 \
  --global_batch_size "$GLOBAL_BATCH_SIZE" \
  --batch_reduction 4 \
  --local_batch_size "$LOCAL_BATCH_SIZE" \
  --max_steps "$MAX_STEPS" \
  --save_every "$SAVE_EVERY" \
  --validate_every "$VALIDATE_EVERY" \
  "$@"
