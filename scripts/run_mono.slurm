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

# Setup Python venv
VENV_DIR="$HOME/venvs/baby-lm-venv"
if [ ! -d "$VENV_DIR" ]; then
    python3.9 -m venv $VENV_DIR
fi
source $VENV_DIR/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Preprocess dataset
echo "Preprocessing /data/MONOLINGUAL/$LANG_CODE..."
python encode_dataset.py --train_path /data/MONOLINGUAL/$LANG_CODE/train \
                         --valid_path /data/MONOLINGUAL/$LANG_CODE/valid
echo "Preprocessing finished."

# Training
cd pretraining
CUDA_VISIBLE_DEVICES=0 python3 train_single_gpu.py \
    --train_path /data/MONOLINGUAL/$LANG_CODE/train \
    --valid_path /data/MONOLINGUAL/$LANG_CODE/valid \
    --tokenizer_path ../tokenizers/tokenizer.json \
    --config_file ../configs/small.json \
    --local_batch_size 32 \
    --global_batch_size 256 \
    --mixed_precision
echo "Training finished."
