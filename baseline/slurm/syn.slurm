#!/bin/bash
#SBATCH --job-name=syn
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
# python syn_test.py --data L2R --csv_dir /dvr/baseline/final_results/SyN/ --label_dir /datasets/L2R_pp/Test/ --output_dir /datasets/L2R_pp/SyN/ --num_labels 4 
python syn_test.py --data CHAOS --csv_dir /dvr/baseline/final_results/SyN/ --label_dir/datasets/CHAOS_pp/Test/ --output_dir /datasets/CHAOS_pp/SyN/ --num_labels 1 
# python syn_test.py --data MRXFDG --csv_dir /dvr/baseline/final_results/SyN/ --label_dir /datasets/MRXFDG_pp/Test/ --output_dir /datasets/MRXFDG_pp/SyN/ --num_labels 22 