#!/bin/bash
#SBATCH -J baby-lm-mono
#SBATCH -A BUTTERY-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH -p ampere

module purge
module load rhel8/default-amp

# Get language argument
LANG_CODE=$1
if [ -z "$LANG_CODE" ]; then
    echo "Error: No language code provided."
    echo "Usage: sbatch run_mono.slurm <lang>"
    exit 1
fi

echo "JobID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "Running on: $(hostname)"
echo "Current directory: $(pwd)"
echo "Language: $LANG_CODE"

# Create and activate virtual environment
python3.9 -m venv /scripts/venvs/demo
source /scripts/venvs/demo/bin/activate
python -m pip install --upgrade pip
pip install "torch>=2.0.0" tqdm numpy tokenizers wandb datasets transformers "protobuf>=3.20.0" scikit-learn

# Preprocess dataset
echo "Preprocessing /data/MONOLINGUAL/$LANG_CODE..."
cd .. || exit 1   # go from /scripts -> root
python preprocessing/updated_preprocess.py \
    --dataset_type monolingual \
    --mono_lang $LANG_CODE \
    --tokenizer tokenizers/tokenizer.json \
    --base_dir data \
    --seq_length 128 \
    --shard_size_bytes 100000000 \
    --batch_size 1000
echo "Preprocessing finished."

# Training
cd pretraining || exit 1
CUDA_VISIBLE_DEVICES=0 python3 train_single_gpu.py \
    --train_path ../data/MONOLINGUAL/$LANG_CODE/train \
    --valid_path ../data/MONOLINGUAL/$LANG_CODE/valid \
    --tokenizer_path ../tokenizers/tokenizer.json \
    --config_file ../configs/small.json \
    --local_batch_size 32 \
    --global_batch_size 256 \
    --mixed_precision
echo "Training finished."

