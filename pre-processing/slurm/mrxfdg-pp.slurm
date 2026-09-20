#!/bin/bash
#SBATCH --job-name=mrxfdg-pp
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=acc
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module unload pytorch
module unload python
module load python/3.10.8-gpu
module load pytorch/1.13.1-gpu

# export ANTSPATH=/ants-2.6.2/
# export PATH=${ANTSPATH}/bin:${PATH}

# Activate preenv with clean environment
source /preenv/bin/activate

# Clean the Python path to avoid conflicts
unset PYTHONPATH

which python
python -c "import sys; print(sys.prefix)"
python -c "import numpy; print(numpy.__version__, numpy.__file__)"

# Verify environment is working
echo "Python executable: $(which python)"

# Run full preprocessing (debug showed environment is working)
echo "Running MRXFDG preprocessing..."
python mrxfdg_pp.py \
    --input_dir /datasets/MRXFDG_og/ \
    --output_dir /datasets/MRXFDG_pp/ \
    --atlas_dir /dvr/pre-processing/atlases/ \
    --skull_dir /dvr/pre-processing/skull/ \
    --max_patients 10