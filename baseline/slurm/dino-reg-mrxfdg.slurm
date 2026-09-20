#!/bin/bash
#SBATCH --job-name=dino-reg-mrxfdg
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --account=acc
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module unload pytorch
module unload python
module load python/3.10.8-gpu
module load pytorch/1.13.1-gpu
module load gcc
module load nvidia/nvhpc

# Activate env
source /dinoregenv/bin/activate

python DINO-Reg/inference_l2rmrct.py \
    --data MRXFDG \
    --csv_dir /datasets/DINO-Reg/ \
    --test_dir /datasets/MRXFDG_pp/Test/ 
