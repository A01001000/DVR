#!/bin/bash
#SBATCH --job-name=vm-test
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --account=tc067-s2749465
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module unload pytorch
module unload python
module load python/3.10.8-gpu
module load pytorch/1.13.1-gpu

# Activate gpuenv
source /work/tc067/tc067/s2749465/vmenv/bin/activate

cd /work/tc067/tc067/s2749465/dvr/baseline/

python nestedmorph/test_clean.py \
        --model VoxelMorph \
        --checkpoint experiments/VoxelMorph_optimized_L2R_multiset/best_model_VoxelMorph.pth.tar \
        --test-dir /work/tc067/tc067/s2749465/datasets/L2R_pp/Test/ \
        --output-dir nestedmorph/test_results_VoxelMorph_L2R \
        --save-images 
    
python nestedmorph/test_clean.py \
    --model VoxelMorph \
    --checkpoint experiments/VoxelMorph_optimized_L2R_multiset/best_model_VoxelMorph.pth.tar \
    --test-dir /work/tc067/tc067/s2749465/datasets/CHAOS_pp/Test/ \
    --output-dir nestedmorph/test_results_VoxelMorph_CHAOS \
    --save-images 
