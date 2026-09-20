#!/bin/bash
#SBATCH --job-name=test-initial
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=acc
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module unload pytorch
module unload python
module load python/3.10.8-gpu
module load pytorch/1.13.1-gpu

# Activate env
source /dinoregenv/bin/activate

python initial_test.py \
    --data CHAOS \
    --csv_dir /dvr/baseline/results/initial/ \
    --label_dir /datasets/CHAOS_pp/Test/ 

python initial_test.py \
    --data L2R \
    --csv_dir /dvr/baseline/results/initial/ \
    --label_dir /datasets/L2R_pp/Test/ 

python initial_test.py \
    --data MRXFDG \
    --csv_dir /dvr/baseline/results/initial/ \
    --label_dir /datasets/MRXFDG_pp/Test/ 