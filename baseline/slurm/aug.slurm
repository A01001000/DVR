#!/bin/bash
#SBATCH --job-name=aug-baseline
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
source dvrenv/bin/activate

python data_aug.py \
   --dir /datasets/MRXFDG_pp/Train/ \
   --aug_dir /datasets/MRXFDG_pp/Train_aug/ \
   --num_augs 10 
