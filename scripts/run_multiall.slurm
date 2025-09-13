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

VENV_DIR="$HOME/venvs/baby-lm-venv"
if [ ! -d "$VENV_DIR" ]; then
    python3.9 -m venv $VENV_DIR
fi
source $VENV_DIR/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Preprocess
echo "Preprocessing /data/MULTILINGUAL-ALL..."
python encode_dataset.py --train_path /data/MULTILINGUAL-ALL/train \
                         --valid_path /data/MULTILINGUAL-ALL/valid
echo "Preprocessing finished."

# Training
cd pretraining
CUDA_VISIBLE_DEVICES=0 python3 train_single_gpu.py \
    --train_path /data/MULTILINGUAL-ALL/train \
    --valid_path /data/MULTILINGUAL-ALL/valid \
    --tokenizer_path ../tokenizers/tokenizer.json \
    --config_file ../configs/small.json \
    --local_batch_size 32 \
    --global_batch_size 256 \
    --mixed_precision
echo "Training finished."
