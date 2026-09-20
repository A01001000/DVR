#!/bin/bash
#SBATCH --job-name=nm-l2r
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
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

# Train NestedMorph with 4 datasets (recommended: 300-500 epochs)
python nestedmorph/train_nifti.py \
    --model NestedMorph \
    --data-dirs /work/tc067/tc067/s2749465/datasets/L2R_pp/Train/ /work/tc067/tc067/s2749465/datasets/CHAOS_pp/Train/ \
    --data-name L2R_multiset \
    --epochs 100 \
    --lr 1e-4 \
    --batch-size 1

# Test on L2R dataset (only if checkpoint exists)
if [ -f "nestedmorph/experiments/NestedMorph_nifti_L2R_multiset/best_model_NestedMorph.pth.tar" ]; then
    python nestedmorph/test_clean.py \
        --model NestedMorph \
        --checkpoint nestedmorph/experiments/NestedMorph_nifti_L2R_multiset/best_model_NestedMorph.pth.tar \
        --test-dir /work/tc067/tc067/s2749465/datasets/L2R_pp/Test/ \
        --output-dir nestedmorph/test_results_L2R \
        --save-images 
    
    # Test on CHAOS dataset
    python nestedmorph/test_clean.py \
        --model NestedMorph \
        --checkpoint nestedmorph/experiments/NestedMorph_nifti_L2R_multiset/best_model_NestedMorph.pth.tar \
        --test-dir /work/tc067/tc067/s2749465/datasets/CHAOS_pp/Test/ \
        --output-dir nestedmorph/test_results_CHAOS \
        --save-images 
else
    echo "Checkpoint not found. Training may have failed or not completed."
fi 