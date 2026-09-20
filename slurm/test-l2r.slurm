#!/bin/bash
#SBATCH --job-name=dvr-l2r-train
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=acc
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module unload pytorch
module unload python
module load python/3.11.5-gpu
module load pytorch/2.2.0-gpu 

# Activate gpuenv
source /dvrenv2/bin/activate

# To fix memory issues with PyTorch; avoids fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# For torch.compile
export TRITON_CACHE_DIR="/triton_cache"
mkdir -p $TRITON_CACHE_DIR

python main2.py \
    --test \
    --data_name L2R \
    --model_name L2R \
    --model_dir /dvr-experiments/ \
    --test_dir /datasets/L2R_pp/Test/ \
    --epochs 150 \
    --steps 5

python main2.py \
    --test \
    --data_name CHAOS \
    --model_name L2R \
    --model_dir /dvr-experiments/ \
    --test_dir /datasets/CHAOS_pp/Test/ \
    --epochs 150 \
    --steps 5