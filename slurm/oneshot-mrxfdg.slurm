#!/bin/bash
#SBATCH --job-name=oneshot-mrxfdg
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=05:00:00
#SBATCH --mem=30000
#SBATCH --cpus-per-task=4
#SBATCH --output=/logs/train_%j.out
#SBATCH --error=/logs/train_%j.err

# Activate gpuenv
source /miniconda3/etc/profile.d/conda.sh
conda activate /miniconda3/envs/dvrenv

# To fix memory issues with PyTorch; avoids fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# For torch.compile
export TRITON_CACHE_DIR="/triton_cache"
mkdir -p $TRITON_CACHE_DIR

PROJECT_DIR="/home/dvr"
SCRATCH_DIR="/disk/scratch/"
mkdir -p $SCRATCH_DIR

echo "Job running on $(hostname)"
echo "Scratch directory: $SCRATCH_DIR"

# Remove any existing files to ensure fresh copies
rm -rf $SCRATCH_DIR/dvr2
rm -rf $SCRATCH_DIR/dvr-experiments
rm -f $SCRATCH_DIR/no_vlearn.py
rm -f $SCRATCH_DIR/main2.py
rm -f $SCRATCH_DIR/clear_cache.py

mkdir -p $SCRATCH_DIR/datasets/MRXFDG_pp/Train
mkdir -p $SCRATCH_DIR/datasets/MRXFDG_pp/Test
cp -r $PROJECT_DIR/datasets/MRXFDG_pp/Train/ $SCRATCH_DIR/datasets/MRXFDG_pp/
cp -r $PROJECT_DIR/datasets/MRXFDG_pp/Test/ $SCRATCH_DIR/datasets/MRXFDG_pp/
cp -r $PROJECT_DIR/dvr2 $SCRATCH_DIR/
cp -r $PROJECT_DIR/dvr-experiments $SCRATCH_DIR/
cp $PROJECT_DIR/no_vlearn.py $SCRATCH_DIR/
cp $PROJECT_DIR/clear_cache.py $SCRATCH_DIR/
cp $PROJECT_DIR/test_device_fix.py $SCRATCH_DIR/
cp $PROJECT_DIR/test_device_fix_quick.py $SCRATCH_DIR/
cp $PROJECT_DIR/test_dino_init.py $SCRATCH_DIR/
cp $PROJECT_DIR/test_brain_features.py $SCRATCH_DIR/
cp $PROJECT_DIR/test_baseline.py $SCRATCH_DIR/
cp $PROJECT_DIR/requirements.txt $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/logs

echo "✅ Fresh code with randomization fixes copied to scratch directory"

cd $SCRATCH_DIR

# Clear any cached DINO features and models to force reprocessing with updated MRXFDG dataset
echo "🧹 Clearing cached data to ensure fresh processing with updated MRXFDG dataset..."
python clear_cache.py --data_name MRXFDG --clear_all --model_dir $SCRATCH_DIR/dvr-experiments/

# Also clear the one-shot specific cache
rm -rf $SCRATCH_DIR/dvr-experiments/Ablation/One-shot/MRXFDG/*
echo "🧹 Also cleared one-shot model cache"

python no_vlearn.py \
    --train \
    --test \
    --data_name MRXFDG \
    --model_name MRXFDG \
    --model_dir $SCRATCH_DIR/dvr-experiments/ \
    --train_dir $SCRATCH_DIR/datasets/MRXFDG_pp/Train/ \
    --test_dir $SCRATCH_DIR/datasets/MRXFDG_pp/Test/ \
    --epochs 100 \
    --dataset_type brain

echo "🔬 Running baseline comparison test after main training..."
python test_baseline.py
    
cp -r dvr-experiments/Ablation/One-shot/MRXFDG $PROJECT_DIR/dvr-experiments/Ablation/One-shot/
cp -r dvr-experiments/Dino-Features $PROJECT_DIR/dvr-experiments/
cp -r logs $PROJECT_DIR/new_logs