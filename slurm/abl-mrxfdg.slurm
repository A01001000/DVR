#!/bin/bash
#SBATCH --job-name=dvr-abl-mrxfdg
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --account=acc
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module unload pytorch
module unload python
module load python/3.11.5-gpu
module load pytorch/2.2.0-gpu 

# Activate gpuenv
source dvrenv2/bin/activate

# To fix memory issues with PyTorch; avoids fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# For torch.compile
export TRITON_CACHE_DIR="/triton_cache"
mkdir -p $TRITON_CACHE_DIR

# Activate gpuenv
source /dvrenv2/bin/activate

python ablation/no_vlearn.py \
    --train \
    --test \
    --data_name MRXFDG \
    --model_name MRXFDG \
    --model_dir /dvr-experiments/ \
    --train_dir /datasets/MRXFDG_pp/Train/ \
    --test_dir /datasets/MRXFDG_pp/Test/ \
    --epochs 100 