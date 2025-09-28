#!/bin/bash
#SBATCH -J baby-lm-multismall-ddp
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
echo "Host: $(hostname)"

# Optional venv setup (commented out; mirror existing style)
# python3.9 -m venv /scripts/venvs/demo
# source /scripts/venvs/demo/bin/activate
# pip install --upgrade pip
# pip install torch tqdm numpy tokenizers wandb datasets transformers protobuf scikit-learn

cd .. || exit 1

CONFIG=configs/multilingual.json \
PREPROCESS_DATASET_TYPE=multilingual_small \
N_GPUS=${N_GPUS:-2} NAME="${NAME:-multismall-ddp}" \
./scripts/run_train_multigpu.sh
