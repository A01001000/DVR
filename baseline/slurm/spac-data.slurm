#!/bin/bash
#SBATCH --job-name=spac-data
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

# Check if directories exist
echo "Checking data directories..."
for dir in "/datasets/L2R_pp/Train/" "/datasets/L2R_pp/Train_aug/" "/datasets/CHAOS_pp/Train/" "/datasets/CHAOS_pp/Train_aug/"; do
    if [ -d "$dir" ]; then
        echo "✓ Found: $dir ($(ls "$dir" | wc -l) items)"
    else
        echo "✗ Missing: $dir"
    fi
done

python SPAC/code/data_util/create_dataset.py \
    --data_roots /datasets/L2R_pp/Train/ /datasets/L2R_pp/Train_aug/ /datasets/CHAOS_pp/Train/ /datasets/CHAOS_pp/Train_aug/ \
    --dataset_names l2r l2r chaos chaos \
    --output_h5 /datasets/SPAC/l2r_train.h5 \
    --output_json /dvr/baseline/SPAC/code/datasets/l2r_train.json \
    --modalities mr ct \
    --image_size 192 192 192 \
    --test_split 0.2

echo "Training dataset creation completed. Checking test directories..."
for dir in "/datasets/L2R_pp/Test/" "/datasets/CHAOS_pp/Test/"; do
    if [ -d "$dir" ]; then
        echo "✓ Found: $dir ($(ls "$dir" | wc -l) items)"
    else
        echo "✗ Missing: $dir"
    fi
done

python SPAC/code/data_util/create_dataset.py \
    --data_roots /datasets/L2R_pp/Test/ /datasets/CHAOS_pp/Test/  \
    --dataset_names l2r chaos \
    --output_h5 /datasets/SPAC/l2r_test.h5 \
    --output_json /dvr/baseline/SPAC/code/datasets/l2r_test.json \
    --modalities mr ct \
    --image_size 192 192 192 \
    --test_split 0.0