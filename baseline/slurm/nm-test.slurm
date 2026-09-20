#!/bin/bash
#SBATCH --job-name=nm-test
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
source /vmenv/bin/activate

python nestedmorph/test_clean.py \
        --model NestedMorph \
        --checkpoint experiments/NestedMorph_optimized_L2R_multiset/best_model_NestedMorph.pth.tar \
        --test-dir /datasets/L2R_pp/Test/ \
        --output-dir nestedmorph/test_results_NestedMorph_L2R \
        --save-images 
    
python nestedmorph/test_clean.py \
    --model NestedMorph \
    --checkpoint experiments/NestedMorph_optimized_L2R_multiset/best_model_NestedMorph.pth.tar \
    --test-dir /datasets/CHAOS_pp/Test/ \
    --output-dir nestedmorph/test_results_NestedMorph_CHAOS \
    --save-images 
    
python nestedmorph/test_clean.py \
        --model NestedMorph \
        --checkpoint experiments/NestedMorph_optimized_MRXFDG_multiset/best_model_NestedMorph.pth.tar \
        --test-dir /datasets/MRXFDG_pp/Test/ \
        --output-dir nestedmorph/test_results_NestedMorph_MRXFDG \
        --save-images 
