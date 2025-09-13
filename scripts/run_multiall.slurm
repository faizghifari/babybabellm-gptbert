#!/bin/bash
#SBATCH -J baby-lm-multiall
#SBATCH -A BUTTERY-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH -p ampere

module purge
module load rhel8/default-amp

echo "JobID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "Running on: $(hostname)"
echo "Current directory: $(pwd)"

# Create and activate virtual environment
python3.9 -m venv /scripts/venvs/demo
source /scripts/venvs/demo/bin/activate
python -m pip install --upgrade pip
pip install "torch>=2.0.0" tqdm numpy tokenizers wandb datasets transformers "protobuf>=3.20.0" scikit-learn

# Go to root
cd .. || exit 1

# Build tokenizer
cd tokenizers || exit 1
python3 tokenizer.py
cd .. || exit 1

# Preprocess
echo "Preprocessing /data/MULTILINGUAL-ALL..."
python preprocessing/updated_preprocess.py \
    --dataset_type multilingual_all \
    --tokenizer tokenizers/tokenizer.json \
    --base_dir data \
    --seq_length 128 \
    --shard_size_bytes 100000000 \
    --batch_size 1000
echo "Preprocessing finished."

# Training
cd pretraining || exit 1
CUDA_VISIBLE_DEVICES=0 python3 train_single_gpu.py \
    --train_path ../data/MULTILINGUAL-ALL/train \
    --valid_path ../data/MULTILINGUAL-ALL/valid \
    --tokenizer_path ../tokenizers/tokenizer.json \
    --config_file ../configs/small.json \
    --local_batch_size 32 \
    --global_batch_size 256 \
    --mixed_precision
echo "Training finished."

