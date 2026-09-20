#!/bin/bash
#SBATCH --job-name=dvr-mrxfdg-test
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --account=acc
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module unload pytorch
module unload python
module load python/3.11.5-gpu
module load pytorch/2.2.0-gpu 

# Activate gpuenv
source /dvrenv2/bin/activate

python main2.py \
    --test \
    --test_data MRXFDG \
    --dino_dir /dvr-experiments/MRXFDG-Dino-Features/ \
    --test_dir /datasets/MRXFDG/Test/ \
    --save_dir /dvr-experiments/MRXFDG/ \
    --epochs 5000 \
    --steps 5