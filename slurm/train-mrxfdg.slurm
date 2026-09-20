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
    --train \
    --data_name MRXFDG \
    --model_name MRXFDG \
    --model_dir /dvr-experiments/ \
    --train_dir /datasets/MRXFDG_pp/Train/ \
    --epochs 150 \
    --steps 5 \
    --resume

python main2.py \
    --test \
    --data_name MRXFDG \
    --model_name MRXFDG \
    --model_dir /dvr-experiments/ \
    --test_dir /datasets/MRXFDG_pp/Test/ \
    --epochs 150 \
    --steps 5
