#!/bin/bash
#SBATCH --job-name=int-l2r
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=10:00:00
#SBATCH --mem=64000
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

PROJECT_DIR="/dvr"
SCRATCH_DIR="/disk/scratch/"
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
cp $PROJECT_DIR/main2.py $SCRATCH_DIR/
cp $PROJECT_DIR/requirements.txt $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/logs

cd $SCRATCH_DIR

python main2.py \
    --train \
    --data_name L2R \
    --model_name L2R \
    --model_dir $SCRATCH_DIR/dvr-experiments/ \
    --train_dir $SCRATCH_DIR/datasets/L2R_pp/Train/ \
    --train_dir2 $SCRATCH_DIR/datasets/CHAOS_pp/Train/ \
    --epochs 150 \
    --steps 5 \
    --dataset_type abdomen

python main2.py \
    --test \
    --data_name L2R \
    --model_name L2R \
    --model_dir $SCRATCH_DIR/dvr-experiments/ \
    --test_dir $SCRATCH_DIR/datasets/L2R_pp/Test/ \
    --epochs 150 \
    --steps 5 \
    --dataset_type abdomen

python main2.py \
    --test \
    --data_name CHAOS \
    --model_name L2R \
    --model_dir $SCRATCH_DIR/dvr-experiments/ \
    --test_dir $SCRATCH_DIR/datasets/CHAOS_pp/Test/ \
    --epochs 150 \
    --steps 5 \
    --dataset_type abdomen

cp -r dvr-experiments/Ablation/Int-layer/L2R $PROJECT_DIR/dvr-experiments/Ablation/Int-layer/
cp -r dvr-experiments/Dino-Features $PROJECT_DIR/dvr-experiments/
cp -r logs $PROJECT_DIR/new_logs

# rm -rf $SCRATCH_DIR/*