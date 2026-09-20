#!/bin/bash
#SBATCH --job-name=dvr-optimized-train
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=acc
#SBATCH --output=logs/train_optimized_%j.out
#SBATCH --error=logs/train_optimized_%j.err

# Clear any previous modules
module purge

# Load required modules
module load python/3.10.8-gpu
module load pytorch/1.13.1-gpu

# Activate the corrected environment
source /dvrenv/bin/activate

echo "Starting DVR training with CORRECTED spatial transformer"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Train the DVR model with optimized parameters
echo "🎯 Training DVR model..."
python train_optimized.py \\
    --train \\
    --train_dir /datasets/L2R_pp/Train/ \\
    --train_dir2 /datasets/CHAOS_pp/Train/ \\
    --dino_dir /dvr-experiments/L2R-Dino-Features/ \\
    --save_dir /dvr-experiments/L2R-Corrected/ \\
    --epochs 300 \\
    --steps 3

echo "✅ Training completed at $(date)"

# Test on L2R dataset
echo "🧪 Testing on L2R dataset..."
python train_optimized.py \\
    --test \\
    --dino \\
    --test_data L2R \\
    --dino_dir /dvr-experiments/L2R-Dino-Features/ \\
    --test_dir /datasets/L2R_pp/Test/ \\
    --save_dir /dvr-experiments/L2R-Corrected/ \\
    --steps 3

# Test on CHAOS dataset (cross-dataset evaluation)
echo "🧪 Testing on CHAOS dataset (cross-dataset)..."
python train_optimized.py \\
    --test \\
    --dino \\
    --test_data CHAOS \\
    --dino_dir /dvr-experiments/CHAOS-Dino-Features/ \\
    --test_dir //datasets/CHAOS_pp/Test/ \\
    --save_dir /dvr-experiments/L2R-Corrected/ \\
    --steps 3

echo "🏁 All tasks completed at $(date)"
echo "📊 Results saved in: /dvr-experiments/L2R-Corrected/"
