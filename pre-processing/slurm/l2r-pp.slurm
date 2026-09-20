#!/bin/bash
#SBATCH --job-name=l2r-pp
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=acc
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module unload pytorch
module unload python
module load python/3.10.8-gpu
module load pytorch/1.13.1-gpu

# Activate gpuenv
source /preenv/bin/activate

python l2r-pp.py \
    --input_dir /datasets/L2R_og/ \
    --output_dir /datasets/L2R_pp/ 