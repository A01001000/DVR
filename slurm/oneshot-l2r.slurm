#!/bin/bash
#SBATCH --job-name=oneshot-l2r
#SBATCH --partition=Teach-LongJobs
#SBATCH --gres=gpu:1g.16gb:1
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
SCRATCH_DIR="/disk/scratch"
mkdir -p $SCRATCH_DIR

echo "Job running on $(hostname)"
echo "Scratch directory: $SCRATCH_DIR"

mkdir -p $SCRATCH_DIR/datasets/CHAOS_pp/Train
mkdir -p $SCRATCH_DIR/datasets/CHAOS_pp/Test
cp -r $PROJECT_DIR/datasets/CHAOS_pp/Train/ $SCRATCH_DIR/datasets/CHAOS_pp/
cp -r $PROJECT_DIR/datasets/CHAOS_pp/Test/ $SCRATCH_DIR/datasets/CHAOS_pp/
mkdir -p $SCRATCH_DIR/datasets/L2R_pp/Train
mkdir -p $SCRATCH_DIR/datasets/L2R_pp/Test
cp -r $PROJECT_DIR/datasets/L2R_pp/Train/ $SCRATCH_DIR/datasets/L2R_pp/
cp -r $PROJECT_DIR/datasets/L2R_pp/Test/ $SCRATCH_DIR/datasets/L2R_pp/
cp -r $PROJECT_DIR/dvr2 $SCRATCH_DIR/
cp -r $PROJECT_DIR/dvr-experiments $SCRATCH_DIR/
cp $PROJECT_DIR/no_vlearn.py $SCRATCH_DIR/
cp $PROJECT_DIR/requirements.txt $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/logs

cd $SCRATCH_DIR

python no_vlearn.py \
    --train \
    --test \
    --data_name L2R \
    --model_name L2R \
    --model_dir $SCRATCH_DIR/dvr-experiments/ \
    --train_dir $SCRATCH_DIR/datasets/L2R_pp/Train/ \
    --train_dir2 $SCRATCH_DIR/datasets/CHAOS_pp/Train/ \
    --test_dir $SCRATCH_DIR/datasets/L2R_pp/Test/ \
    --epochs 100

cp -r dvr-experiments/Ablation/One-shot/L2R $PROJECT_DIR/dvr-experiments/Ablation/One-shot/
cp -r dvr-experiments/Dino-Features $PROJECT_DIR/dvr-experiments/
cp -r logs $PROJECT_DIR/new_logs