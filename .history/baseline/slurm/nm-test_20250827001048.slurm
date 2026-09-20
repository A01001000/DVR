#!/bin/bash
#SBATCH --job-name=nm-mrxfdg
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --account=tc067-s2749465
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module unload pytorch
module unload python
module load python/3.10.8-gpu
module load pytorch/1.13.1-gpu

# Activate vmenv
source /work/tc067/tc067/s2749465/vmenv/bin/activate

cd /work/tc067/tc067/s2749465/dvr/baseline/

# Train NestedMorph with OPTIMIZED settings
python nestedmorph/train_optimized.py \
    --model NestedMorph \
    --data-dirs /work/tc067/tc067/s2749465/datasets/MRXFDG_pp/Train/ \
    --data-name MRXFDG_multiset \
    --epochs 120 \
    --lr 1e-4 \
    --batch-size 1 \
    --img-size 128 128 128 \
    --patience 20

echo "Optimized NestedMorph training completed!"
