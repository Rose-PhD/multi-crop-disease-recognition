#!/bin/bash
#SBATCH -p dmb                 # partition
#SBATCH --gres=gpu:1           # request 1 GPU
#SBATCH --cpus-per-task=4      # CPU cores
#SBATCH --mem=32G              # memory
#SBATCH -t 24:00:00            # time limit
#SBATCH -J train_resnet18      # job name

#SBATCH --output=/home/nalwangar/wilfred/multi-crop-disease-recognition/logs/slurm_flat_out_%j.out
#SBATCH --error=/home/nalwangar/wilfred/multi-crop-disease-recognition/logs/slurm_flat_err_%j.err

echo "=== Job started on $(hostname) at $(date) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION ==="
echo "=== SLURM_NODELIST: $SLURM_NODELIST ==="
echo "=== GPUs assigned: $CUDA_VISIBLE_DEVICES ==="

# -------------------------------------------------
# Load your environment correctly
# -------------------------------------------------
source /deepstore/software/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate dev-multicrop

PYTHON_BIN=$(which python)
if [[ "$PYTHON_BIN" != *"dev-multicrop"* ]]; then
    echo "ERROR: conda activate dev-multicrop failed. Python resolves to: $PYTHON_BIN"
    exit 1
fi

echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# Move to your working directory (recommended)
cd /home/nalwangar/wilfred/multi-crop-disease-recognition || { echo "ERROR: working directory not found"; exit 1; }
echo "Working directory: $(pwd)"

# -------------------------------------------------
# Run your training script
# -------------------------------------------------
python -u -m flat.train_flat

echo "=== Training finished at $(date) ==="

# -------------------------------------------------
# Run testing (only if training succeeded)
# -------------------------------------------------
if [ $? -eq 0 ]; then
    echo "=== Starting flat testing at $(date) ==="
    python -u -m flat.test_flat
    echo "=== Testing finished at $(date) ==="
else
    echo "=== Training failed — skipping testing ==="
    exit 1
fi

echo "=== Job finished at $(date) ==="



