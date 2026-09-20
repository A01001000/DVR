#!/bin/bash
#SBATCH --job-name=spac-mrxfdg
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --account=acc
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module unload pytorch
module unload python
module load python/3.10.8-gpu
module load pytorch/1.13.1-gpu

# Activate gpuenv
source /spacenv/bin/activate

unset PYTHONPATH

which python
python -c "import torch; print(torch.__file__)"
python -c "import torchvision; print(torchvision.__file__)"
python -c "import sys; print(sys.prefix)"
python -c "import numpy; print(numpy.__version__, numpy.__file__)"

# Verify environment is working
echo "Python executable: $(which python)"

python -u SPAC/code/main.py \
    --data_type mrxfdg_train \
    --train_data /datasets/SPAC/mrxfdg_train.h5 

python -u SPAC/code/test.py \
    --data_type mrxfdg_test \
    --test_data /datasets/SPAC/mrxfdg_test.h5 \
    --save_warped 