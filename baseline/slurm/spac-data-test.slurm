#!/bin/bash
#SBATCH --job-name=spac-test-data
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=acc
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module unload pytorch
module unload python
module load python/3.10.8-gpu
module load pytorch/1.13.1-gpu

# Activate gpuenv
source /preenv/bin/activate

python SPAC/code/data_util/create_dataset.py \
    --data_root /datasets/L2R_pp/Test/ \
    --dataset_name l2r \
    --output_h5 /datasets/SPAC/l2r_test.h5 \
    --output_json /dvr/baseline/SPAC/code/datasets/l2r_test.json \
    --modalities mr ct \
    --image_size 192 192 192 \
    --test_split 0.0

python SPAC/code/data_util/create_dataset.py \
    --data_root /datasets/CHAOS_pp/Test/  \
    --dataset_name chaos \
    --output_h5 /datasets/SPAC/chaos_test.h5 \
    --output_json /dvr/baseline/SPAC/code/datasets/chaos_test.json \
    --modalities mr ct \
    --image_size 192 192 192 \
    --test_split 0.0