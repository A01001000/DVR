#!/bin/bash
#SBATCH --job-name=raw-mrxfdg
#SBATCH --partition=Teach-LongJobs
#SBATCH --gres=gpu:1g.16gb:1
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

PROJECT_DIR="/home/dvr"
SCRATCH_DIR="/disk/scratch"
mkdir -p $SCRATCH_DIR

echo "Job running on $(hostname)"
echo "Scratch directory: $SCRATCH_DIR"

rm -rf $SCRATCH_DIR/datasets/MRXFDG_pp/
mkdir -p $SCRATCH_DIR/datasets/MRXFDG_pp/Train
mkdir -p $SCRATCH_DIR/datasets/MRXFDG_pp/Test
cp -r $PROJECT_DIR/datasets/MRXFDG_pp/Train/ $SCRATCH_DIR/datasets/MRXFDG_pp/
cp -r $PROJECT_DIR/datasets/MRXFDG_pp/Test/ $SCRATCH_DIR/datasets/MRXFDG_pp/
rsync -av --exclude='.git' $PROJECT_DIR/dvr2 $SCRATCH_DIR/
cp -r $PROJECT_DIR/dvr-experiments $SCRATCH_DIR/
cp $PROJECT_DIR/main2.py $SCRATCH_DIR/
cp $PROJECT_DIR/requirements.txt $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/logs

cd $SCRATCH_DIR

python main2.py \
    --train \
    --data_name MRXFDG \
    --model_name MRXFDG \
    --model_dir $SCRATCH_DIR/dvr-experiments/ \
    --train_dir $SCRATCH_DIR/datasets/MRXFDG_pp/Train/ \
    --epochs 150 \
    --steps 5 \
    --use-raw-images

python main2.py \
    --test \
    --data_name MRXFDG \
    --model_name MRXFDG \
    --model_dir $SCRATCH_DIR/dvr-experiments/ \
    --test_dir $SCRATCH_DIR/datasets/MRXFDG_pp/Test/ \
    --epochs 150 \
    --steps 5 \
    --use-raw-images

cp -r dvr-experiments/Ablation/Raw-img/MRXFDG $PROJECT_DIR/dvr-experiments/Ablation/Raw-img/MRXFDG
cp -r dvr-experiments/Dino-Features $PROJECT_DIR/dvr-experiments/
cp -r logs $PROJECT_DIR/new_logs