#!/bin/bash
#SBATCH -J baby-lm-mono-ddp
#SBATCH -A BUTTERY-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH -p ampere

module purge
module load rhel8/default-amp

LANG_CODE=$1
if [ -z "$LANG_CODE" ]; then
    echo "Error: No language code provided."
    echo "Usage: sbatch scripts/run_multigpu_mono.slurm <lang>"
    exit 1
fi

echo "JobID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "Host: $(hostname)"

# Optional: venv setup (uncomment if needed)
# python3.9 -m venv /scripts/venvs/demo
# source /scripts/venvs/demo/bin/activate
# pip install --upgrade pip
# pip install torch tqdm numpy tokenizers wandb datasets transformers protobuf scikit-learn

cd .. || exit 1

# Multi-GPU training; shards/tokenizer will be created if missing
CONFIG=configs/base.json \
PREPROCESS_DATASET_TYPE=monolingual PREPROCESS_MONO_LANG=$LANG_CODE \
N_GPUS=${N_GPUS:-2} NAME="${NAME:-mono-$LANG_CODE-ddp}" \
./scripts/run_train_multigpu.sh
