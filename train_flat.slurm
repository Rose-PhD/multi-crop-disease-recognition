#!/bin/bash
#SBATCH -p dmb                 # partition
#SBATCH --gres=gpu:1           # request 1 GPU
#SBATCH --cpus-per-task=4      # CPU cores
#SBATCH --mem=32G              # memory
#SBATCH -t 24:00:00            # time limit
#SBATCH -J train_resnet18      # job name

#SBATCH --output=/home/nalwangar/wilfred/logs_hierY/slurm_train_out_%j.out
#SBATCH --error=/home/nalwangar/wilfred/logs_hierY/slurm_train_err_%j.err

echo "=== Job started on $(hostname) at $(date) ==="

# -------------------------------------------------
# Load your environment correctly
# -------------------------------------------------
source ~/.bashrc      # IMPORTANT: enables conda
conda activate dev-multicrop

echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# Move to your working directory (recommended)
cd /home/nalwangar/wilfred

# -------------------------------------------------
# Run your training script
# -------------------------------------------------
python train_flat.py

echo "=== Job finished at $(date) ==="



