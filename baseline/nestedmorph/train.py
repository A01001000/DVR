import sys
from scripts.train import train_model
from scripts.train_cyclemorph import train_model as train_cyclemorph
from src.utils.config import device
import argparse

# Set paths and training configuration
t1_dir = "../MRXFDG_voxelmorph/MR"       # e.g., "./data/t1/"
dwi_dir = "../MRXFDG_voxelmorph/CT"     # e.g., "./data/dwi/"

model_label = "VoxelMorph"

# Optional config
epochs = 300
img_size = "192,192,192" #TODO: CHANGE DEPENDING ON DATASET
lr = 0.0001
batch_size = 4
cont_training = False

sys.argv = ["train.py",
            "--t1_dir", t1_dir,
            "--dwi_dir", dwi_dir,
            "--model_label", model_label,
            "--epochs", str(epochs),
            "--img_size", img_size,
            "--lr", str(lr),
            "--batch_size", str(batch_size)]

if not cont_training:
    sys.argv.append("--cont_training")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t1_dir', type=str, required=True)
    parser.add_argument('--dwi_dir', type=str, required=True)
    parser.add_argument('--model_label', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--img_size', type=str, default="64,64,64")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--cont_training', action='store_false')
    return parser.parse_args()

args = parse_args()

# Choose correct training function
train_fn = train_cyclemorph if args.model_label.lower() == "cyclemorph" else train_model

# Run training
# TODO: Check if model is saved / deformation field / loss logs (jacobian)
train_fn(
    t1_dir=args.t1_dir,
    dwi_dir=args.dwi_dir,
    model_label=args.model_label,
    epochs=args.epochs,
    img_size=args.img_size,
    lr=args.lr,
    batch_size=args.batch_size,
    cont_training=args.cont_training,
    device=device
)
