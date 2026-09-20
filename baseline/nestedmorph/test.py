import os
import glob
import csv
import time
import torch
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import DataLoader
from monai.transforms import Compose
from medpy.metric import binary, dc, hd95
from torchvision import transforms
from scipy.ndimage import binary_erosion

from src.models.voxelmorph import VoxelMorph
from src.models.nestedmorph import NestedMorph
from src.losses.losses import MINDLoss, Grad3d
from src.data.datasets import RegistrationDataset
from src.data.test_datasets import TestRegistrationDataset
from src.utils.utils import register_model, dice
from src.utils.config import device
from src.data.trans import *

# Compute Dice Score
def compute_dice(pred, target):
    """
    Compute Dice coefficient between prediction and target.
    Args:
        pred (torch.Tensor): Predicted segmentation
        target (torch.Tensor): Ground truth segmentation
    Returns:
        float: Dice score
    """
    pred_np = (pred.detach().cpu().numpy() > 0.5).astype(np.uint8)
    target_np = (target.detach().cpu().numpy() > 0.5).astype(np.uint8)

    # Compute mean Dice across batch
    dice_scores = []
    for p, t in zip(pred_np, target_np):
        intersection = np.sum(p * t)
        total = np.sum(p) + np.sum(t)
        if total == 0:
            dice_scores.append(1.0)  # Perfect score if both empty
        else:
            dice_scores.append(2.0 * intersection / total)
    
    return np.mean(dice_scores)

def compute_dice_coefficient(mask1, mask2):
    """
    Compute the Dice Similarity Coefficient between two binary masks.
    """
    intersection = np.sum(mask1 * mask2)
    size1 = np.sum(mask1)
    size2 = np.sum(mask2)
    if size1 + size2 == 0:
        return 1.0  # Perfect score if both empty
    return (2.0 * intersection) / (size1 + size2)

def compute_label_wise_dice(seg1, seg2, labels):
    """Compute Dice for each label/organ."""
    dice_results = []
    for label in labels:
        # Isolate current label in both segmentations
        seg1_label = (seg1 == label).astype(np.uint8)
        seg2_label = (seg2 == label).astype(np.uint8)
        
        dice = compute_dice_coefficient(seg1_label, seg2_label)
        dice_results.append(dice)

    return dice_results

def save_warped_image(warped_img, filename, affine=None):
    """Save warped image as NIfTI file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if affine is None:
        affine = np.eye(4)  # Identity affine matrix
    
    # Convert to numpy if tensor
    if torch.is_tensor(warped_img):
        warped_img = warped_img.detach().cpu().numpy()
    
    # Remove batch dimension if present
    if len(warped_img.shape) == 4 and warped_img.shape[0] == 1:
        warped_img = warped_img[0]
    
    # Create NIfTI image and save
    nii_img = nib.Nifti1Image(warped_img, affine)
    nib.save(nii_img, filename)

def save_metrics_csv(metrics, output_path):
    """Save evaluation metrics to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])  # header
        for key, value in metrics.items():
            writer.writerow([key, value])

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test VoxelMorph/NestedMorph models')
    parser.add_argument('--model', type=str, default='VoxelMorph', 
                        choices=['VoxelMorph', 'NestedMorph'],
                        help='Model to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--output-dir', type=str, default='./test_results',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--img-size', type=int, nargs=3, default=[192, 192, 192],
                        help='Image size (H W D)')
    parser.add_argument('--save-images', action='store_true',
                        help='Save warped images and deformation fields')
    
    return parser.parse_args()

# Main evaluation function
def evaluate_model():
    """Main evaluation function."""
    # For now, use hardcoded paths - you can replace with argparse later
    model_label = "VoxelMorph"  # Change to "NestedMorph" as needed
    img_size = (192, 192, 192)
    num_channels = 4
    label_list = list(range(1, num_channels + 1))
    batch_size = 1
    
    # Update paths to your actual data
    output_dir = "./test_results"
    ckpt_path = "./experiments/VoxelMorph_1_mind_1_diffusion_1_L2R/final_model_VoxelMorph.pth.tar"
    
    # Data directories
    moving_test_dir = "../datasets/L2R_voxelmorph/Test/MR/"
    fixed_test_dir = "../datasets/L2R_voxelmorph/Test/CT/"
    moving_label_dir = "../datasets/L2R_labels_voxelmorph/Test/MR/"
    fixed_label_dir = "../datasets/L2R_labels_voxelmorph/Test/CT/"
    
    # Create output directories
    os.makedirs(f"{output_dir}/flows", exist_ok=True)
    os.makedirs(f"{output_dir}/warped_images", exist_ok=True)
    os.makedirs(f"{output_dir}/warped_segmentations", exist_ok=True)

    # Load model
    if model_label == "VoxelMorph":
        model = VoxelMorph(
            input_dim=2,  # 2-channel input (moving+fixed)
            enc_nf=[16, 32, 32, 32],
            dec_nf=[32, 32, 32, 32, 32, 16, 16],
            int_steps=7
        ).to(device)
    elif model_label == "NestedMorph":
        model = NestedMorph().to(device)
    else:
        raise ValueError(f"Unknown model: {model_label}")
    
    # Load checkpoint
    print(f"Loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # Load spatial transformer for warping
    reg_model = register_model(img_size, 'nearest').to(device)
    reg_model.eval()

    # Data transforms
    test_composed = transforms.Compose([
        NumpyType((np.float32, np.float32)),
    ])

    # Build test dataset
    t1_files = glob.glob(moving_test_dir + '*.pkl')
    dwi_files = glob.glob(fixed_test_dir + '*.pkl')
    t1_label_files = glob.glob(moving_label_dir + '*.pkl')
    dwi_label_files = glob.glob(fixed_label_dir + '*.pkl')
    
    print(f"Found {len(t1_files)} moving images and {len(dwi_files)} fixed images")
    
    # Pair files
    main_dict = {os.path.basename(f).split('_')[0]: f for f in dwi_files}
    paired_files = [(t1_file, main_dict.get(os.path.basename(t1_file).split('_')[0]))
                    for t1_file in t1_files 
                    if os.path.basename(t1_file).split('_')[0] in main_dict]
    
    main_label_dict = {os.path.basename(f).split('_')[0]: f for f in dwi_label_files}
    paired_label_files = [(t1_file, main_label_dict.get(os.path.basename(t1_file).split('_')[0]))
                    for t1_file in t1_label_files
                    if os.path.basename(t1_file).split('_')[0] in main_label_dict]

    print(f"Found {len(paired_files)} paired image files")
    print(f"Found {len(paired_label_files)} paired label files")

    # Create datasets
    test_set = RegistrationDataset(
        [pair[0] for pair in paired_files],
        [pair[1] for pair in paired_files],
        transforms=test_composed,
        img_size=img_size
    )
    
    label_test_set = TestRegistrationDataset(
        [pair[0] for pair in paired_label_files],
        [pair[1] for pair in paired_label_files],
        transforms=test_composed,
        img_size=img_size, 
        num_classes=num_channels
    )

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_label_loader = DataLoader(label_test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize metric containers
    all_dice = []
    all_hd95 = []
    all_jac = []
    all_time = []
    
    print(f"Starting evaluation on {len(test_loader)} test samples...")

    with torch.no_grad():
        for batch_idx, (data, seg_data) in enumerate(zip(test_loader, test_label_loader)): 
            print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
            
            # Move data to device
            data = [d.to(device) for d in data]
            seg_data = [d.to(device) for d in seg_data]
            x, y = data[0], data[1]  # moving, fixed images
            x_seg, y_seg = seg_data[0], seg_data[1]  # moving, fixed segmentations
            
            # Measure inference time
            start_time = time.time()
            output = model((x, y))
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            
            # Extract flow and warped image from model output
            if isinstance(output, tuple):
                warped_image, flow = output
            else:
                # If model only returns flow, warp the image manually
                flow = output
                warped_image = reg_model(x, flow)
            
            # Save deformation field
            flow_np = flow.cpu().numpy()[0]  # Remove batch dimension
            np.save(f"{output_dir}/flows/flow_{batch_idx:04d}.npy", flow_np)
            
            # Save warped image
            save_warped_image(warped_image[0], f"{output_dir}/warped_images/warped_{batch_idx:04d}.nii.gz")
            
            # Warp segmentation
            warped_seg_list = []
            for channel_idx in range(x_seg.shape[1]):  # Iterate over channels
                seg_channel = x_seg[:, channel_idx:channel_idx+1]  # Keep channel dimension
                warped_channel = reg_model(seg_channel.float(), flow)
                warped_seg_list.append(warped_channel)
            
            warped_seg_oh = torch.cat(warped_seg_list, dim=1)  # Concatenate channels
            
            # Convert to hard labels
            warped_seg = torch.argmax(warped_seg_oh, dim=1)  # Remove channel dimension
            
            # Save warped segmentation
            save_warped_image(warped_seg[0], f"{output_dir}/warped_segmentations/warped_seg_{batch_idx:04d}.nii.gz")
            
            # Compute Jacobian determinant ratio
            jac_ratio = compute_non_positive_jacobian_ratio(flow)
            all_jac.append(jac_ratio)
            all_time.append(elapsed_time / x.shape[0])  # Per sample time
            
            # Process each sample in batch
            for sample_idx in range(warped_seg.size(0)):
                pred_seg = warped_seg[sample_idx].cpu().numpy()
                gt_seg = torch.argmax(y_seg[sample_idx], dim=0).cpu().numpy()
                
                # Compute metrics
                dice_scores = compute_label_wise_dice(pred_seg, gt_seg, label_list)
                hd95_scores = compute_label_wise_95hd(pred_seg, gt_seg, label_list)
                
                all_dice.append(dice_scores)
                all_hd95.append(hd95_scores)
                
                print(f"  Sample {sample_idx}: Dice={np.nanmean(dice_scores):.4f}, "
                      f"HD95={np.nanmean(hd95_scores):.2f}, Jac≤0={jac_ratio:.4%}")

    # Calculate final metrics
    all_dice = np.array(all_dice)
    all_hd95 = np.array(all_hd95)
    
    # Overall metrics
    final_dice = np.nanmean(all_dice)
    final_dice_std = np.nanstd(all_dice)
    
    # Per-organ metrics
    final_dice_organ = np.nanmean(all_dice, axis=0)
    final_dice_organ_std = np.nanstd(all_dice, axis=0)
    
    # HD95 metrics (flatten and filter valid values)
    flat_hd95 = all_hd95.flatten()
    valid_hd95 = flat_hd95[~np.isnan(flat_hd95)]
    
    if len(valid_hd95) > 0:
        final_hd95 = np.mean(valid_hd95)
        final_hd95_std = np.std(valid_hd95)
    else:
        print("Warning: No valid HD95 values found.")
        final_hd95 = np.nan
        final_hd95_std = np.nan
    
    # Jacobian metrics
    final_jac = np.mean(all_jac)
    final_jac_std = np.std(all_jac)
    
    # Timing metrics
    final_time = np.mean(all_time)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Final Evaluation Results for {model_label}")
    print(f"{'='*50}")
    print(f"Overall Dice Score    : {final_dice:.4f} ± {final_dice_std:.4f}")
    print(f"Per-organ Dice        : {final_dice_organ}")
    print(f"Per-organ Dice Std    : {final_dice_organ_std}")
    print(f"HD95                  : {final_hd95:.2f} ± {final_hd95_std:.2f}")
    print(f"Jacobian ≤ 0         : {final_jac:.4%} ± {final_jac_std:.4%}")
    print(f"Inference Time        : {final_time:.4f} sec/sample")
    print(f"Number of Parameters  : {n_params / 1e6:.2f}M")
    
    # Save results to CSV
    csv_path = f'{output_dir}/{model_label.lower()}_evaluation_results.csv'
    save_metrics_csv({
        'Model': model_label,
        'Overall_Dice_Score': final_dice,
        'Overall_Dice_Std': final_dice_std,
        'HD95': final_hd95,
        'HD95_Std': final_hd95_std,
        'Jacobian_NonPositive_Ratio': final_jac,
        'Jacobian_NonPositive_Std': final_jac_std,
        'Inference_Time_sec_per_sample': final_time,
        'Number_of_Parameters_M': n_params / 1e6,
        **{f'Dice_Organ_{i}': dice for i, dice in enumerate(final_dice_organ, 1)},
        **{f'Dice_Organ_{i}_Std': std for i, std in enumerate(final_dice_organ_std, 1)}
    }, csv_path)
    
    print(f"\nResults saved to: {csv_path}")
    print(f"Deformation fields saved to: {output_dir}/flows/")
    print(f"Warped images saved to: {output_dir}/warped_images/")
    print(f"Warped segmentations saved to: {output_dir}/warped_segmentations/")

if __name__ == "__main__":
    evaluate_model()

# Compute HD95
def compute_hd95(pred, target):
    """
    Compute 95th percentile Hausdorff Distance.
    Args:
        pred (torch.Tensor): Predicted segmentation
        target (torch.Tensor): Ground truth segmentation
    Returns:
        float: HD95 score
    """
    pred_np = (pred.detach().cpu().numpy() > 0.5).astype(np.uint8)
    target_np = (target.detach().cpu().numpy() > 0.5).astype(np.uint8)

    hd95_scores = []
    for p, t in zip(pred_np, target_np):
        try:
            if np.sum(p) == 0 or np.sum(t) == 0:
                hd95_scores.append(np.nan)  # No surface available
            else:
                hd = hd95(p[0], t[0])
                hd95_scores.append(hd)
        except Exception as e:
            print(f"HD95 computation failed: {e}")
            hd95_scores.append(np.nan)
    
    return np.nanmean(hd95_scores)

def compute_95_hausdorff_distance(mask1, mask2):
    """Compute 95th percentile Hausdorff distance."""
    mask1 = np.squeeze(mask1.astype(bool))
    mask2 = np.squeeze(mask2.astype(bool))
    
    if mask1.shape != mask2.shape:
        raise ValueError(f"Shape mismatch: {mask1.shape} vs {mask2.shape}")
    
    if not np.any(mask1) or not np.any(mask2):
        return np.nan  # No surface points available
        
    try:
        return hd95(mask1, mask2)
    except Exception as e:
        print(f"HD95 computation failed: {e}")
        return np.nan

def compute_label_wise_95hd(seg1, seg2, labels):
    """Compute HD95 for each label/organ."""
    hd95_results = []
    for label in labels:
        seg1_label = (seg1 == label)
        seg2_label = (seg2 == label)

        # Skip if either label is missing
        if not np.any(seg1_label) or not np.any(seg2_label):
            print(f"Label {label} missing in prediction or GT -- skipping HD95.")
            hd95_results.append(np.nan)
            continue

        hd = compute_95_hausdorff_distance(seg1_label, seg2_label)
        hd95_results.append(hd)

    return hd95_results

# Compute Non-positive Jacobian Determinant
def compute_non_positive_jacobian_ratio(flow):
    """
    Compute ratio of voxels with det(Jacobian) ≤ 0.
    Args:
        flow: torch.Tensor (B, 3, H, W, D), deformation field
    Returns:
        float: mean ratio of voxels with det(Jacobian) ≤ 0
    """
    flow_np = flow.detach().cpu().numpy()  # shape (B, 3, H, W, D)
    ratios = []

    for b in range(flow_np.shape[0]):
        phi = flow_np[b]  # shape: (3, H, W, D)

        # Compute gradients of each component of the deformation field
        dFx_dx = np.gradient(phi[0], axis=0)
        dFx_dy = np.gradient(phi[0], axis=1)
        dFx_dz = np.gradient(phi[0], axis=2)

        dFy_dx = np.gradient(phi[1], axis=0)
        dFy_dy = np.gradient(phi[1], axis=1)
        dFy_dz = np.gradient(phi[1], axis=2)

        dFz_dx = np.gradient(phi[2], axis=0)
        dFz_dy = np.gradient(phi[2], axis=1)
        dFz_dz = np.gradient(phi[2], axis=2)

        # Create Jacobian matrix for each voxel
        H, W, D = phi.shape[1:]
        J = np.zeros((H, W, D, 3, 3))

        # Add identity to get deformation gradient (I + ∇u)
        J[..., 0, 0] = 1 + dFx_dx
        J[..., 0, 1] = dFx_dy
        J[..., 0, 2] = dFx_dz

        J[..., 1, 0] = dFy_dx
        J[..., 1, 1] = 1 + dFy_dy
        J[..., 1, 2] = dFy_dz

        J[..., 2, 0] = dFz_dx
        J[..., 2, 1] = dFz_dy
        J[..., 2, 2] = 1 + dFz_dz

        # Compute determinant
        jac_det = np.linalg.det(J)  # shape: (H, W, D)

        # Count non-positive determinants
        non_pos_ratio = np.mean(jac_det <= 0)
        ratios.append(non_pos_ratio)

    return np.mean(ratios)

def compute_dice_coefficient(mask1, mask2):
    """
    Compute the Dice Similarity Coefficient between two binary masks.
    """
    intersection = np.sum(mask1 * mask2)
    size1 = np.sum(mask1)
    size2 = np.sum(mask2)
    return (2.0 * intersection + 1e-5 ) / (size1 + size2 + 1e-5)

def compute_label_wise_dice(seg1, seg2, labels):
    dice_results = []
    for label in labels:
        # Isolate current label in both segmentations
        seg1_label = seg1 == label
        seg2_label = seg2 == label

        # Compute 95% HD for the current label
        dice = compute_dice_coefficient(seg1_label, seg2_label)
        dice_results.append(dice)

    return dice_results

def extract_surface_points(seg):
    binary_seg = seg > 0
    eroded = binary_erosion(binary_seg)
    surface = binary_seg ^ eroded
    return np.array(np.where(surface)).T

def compute_95_hausdorff_distance(mask1, mask2):
    mask1 = np.squeeze(mask1.astype(bool))
    mask2 = np.squeeze(mask2.astype(bool))
    
    if mask1.shape != mask2.shape:
        print("HD95 input shape:", mask1.shape, mask2.shape)
        raise ValueError(f"Shape mismatch: {mask1.shape} vs {mask2.shape}")
        
    return hd95(mask1.astype(bool), mask2.astype(bool))

def compute_label_wise_95hd(seg1, seg2, labels):
    hd95_results = []
    for label in labels:
        seg1_label = (seg1 == label)
        seg2_label = (seg2 == label)

        # Skip if either label is missing
        if not np.any(seg1_label) or not np.any(seg2_label):
            print(f"Label {label} missing in prediction or GT -- skipping.")
            hd95_results.append(np.nan)  # or some sentinel value
            continue

        hd = compute_95_hausdorff_distance(seg1_label, seg2_label)
        hd95_results.append(hd)

    return hd95_results

def score_case(seg_fixed, seg_moving, label_list):
    dice_coefficient = compute_label_wise_dice(seg_fixed, seg_moving, label_list)
    dice_coefficient = np.array(dice_coefficient)

    hd95_value = compute_label_wise_95hd(seg_fixed, seg_moving, label_list)
    hd95_value = np.array(hd95)

    return {'DICE': dice_coefficient,
            'HD95': hd95_value}

def save_metrics_csv(metrics, output_path):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])  # header
        for key, value in metrics.items():
            writer.writerow([key, value])

def flow_to_grid(flow):
    """
    Convert a flow field to a normalized grid for grid_sample.

    Args:
        flow: Tensor of shape (B, 3, H, W, D), flow in voxel displacements
    Returns:
        new_grid: Tensor of shape (B, H, W, D, 3), normalized coordinates for grid_sample
    """
    B, C, H, W, D = flow.shape
    assert C == 3, "Flow must have 3 channels"

    # Create mesh grid in H, W, D order
    h = torch.linspace(-1, 1, H, device=flow.device)
    w = torch.linspace(-1, 1, W, device=flow.device)
    d = torch.linspace(-1, 1, D, device=flow.device)

    # meshgrid: shape (H, W, D)
    meshy, meshw, meshd = torch.meshgrid(h, w, d, indexing='ij')

    # Stack to get grid of shape (H, W, D, 3)
    # Note order: (x, y, z) = (w, h, d)
    grid = torch.stack((meshw, meshy, meshd), dim=-1)  # (H, W, D, 3)

    # Add batch dim
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # (B, H, W, D, 3)

    # Normalize flow: flow channels correspond to (x, y, z) = (W, H, D)
    norm_factor = torch.tensor([W / 2, H / 2, D / 2], device=flow.device).view(1, 3, 1, 1, 1)
    norm_flow = flow / norm_factor  # normalize displacement to [-1, 1]

    # Permute flow to (B, H, W, D, 3) to match grid shape
    norm_flow = norm_flow.permute(0, 2, 3, 4, 1)

    # Add normalized flow to base grid
    new_grid = grid + norm_flow  # (B, H, W, D, 3)

    return new_grid

# L2R Labels
labels = {
    "liver": 1,
    "spleen": 2,
    "right_kidney": 3,
    "left_kidney": 4,
}

# Configuration
model_label = "VoxelMorph"
img_size = (192, 192, 192)
num_channels = 4
label_list = list(range(1, num_channels + 1))
batch_size = 1
logs_folder = "./logs/VoxelMorph_1_mind_1_diffusion_1_L2R/"
ckpt_path = "./experiments/VoxelMorph_1_mind_1_diffusion_1_L2R/final_model_VoxelMorph.pth.tar"
moving_test_dir = "../datasets/L2R_voxelmorph/Test/MR/"
fixed_test_dir = "../datasets/L2R_voxelmorph/Test/CT/"
moving_label_dir = "../datasets/L2R_labels_voxelmorph/Test/MR/"
fixed_label_dir = "../datasets/L2R_labels_voxelmorph/Test/CT/"
os.makedirs("logs/VoxelMorph_1_mind_1_diffusion_1_L2R/flows", exist_ok=True)
os.makedirs("logs/VoxelMorph_1_mind_1_diffusion_1_L2R/pairs", exist_ok=True)

# Load model
model = VoxelMorph(img_size).to(device)
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Count parameters
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Load spatial transformer (for warping)
reg_model = register_model(img_size, 'nearest').to(device)
reg_model.eval()

test_composed = transforms.Compose([
        # RandomFlip(0),  # Randomly flip images
        NumpyType((np.float32, np.float32)),  # Convert images to numpy arrays
    ])

# Build test dataset
t1_files = glob.glob(moving_test_dir + '*.pkl')
dwi_files = glob.glob(fixed_test_dir + '*.pkl')
t1_label_files = glob.glob(moving_label_dir + '*.pkl')
dwi_label_files = glob.glob(fixed_label_dir + '*.pkl')
main_dict = {os.path.basename(f).split('_')[0]: f for f in dwi_files}
paired_files = [(t1_file, main_dict.get(os.path.basename(t1_file).split('_')[0]))
                for t1_file in t1_files 
                if os.path.basename(t1_file).split('_')[0] in main_dict]
main_label_dict = {os.path.basename(f).split('_')[0]: f for f in dwi_label_files}
paired_label_files = [(t1_file, main_label_dict.get(os.path.basename(t1_file).split('_')[0]))
                for t1_file in t1_label_files
                if os.path.basename(t1_file).split('_')[0] in main_label_dict]

# Use full list as test set
test_set = RegistrationDataset(
    [pair[0] for pair in paired_files],
    [pair[1] for pair in paired_files],
    transforms=test_composed,  # No random transforms
    img_size=img_size
)
label_test_set = TestRegistrationDataset(
    [pair[0] for pair in paired_label_files],
    [pair[1] for pair in paired_label_files],
    transforms=test_composed,  # No random transforms
    img_size=img_size, 
    num_classes=num_channels
)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
test_label_loader = DataLoader(label_test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

all_dice = []
all_hd95 = []
all_jac = []
all_time = []
# all_dice_per_organ = []
i = 0

with torch.no_grad():
    for data, seg_data in zip(test_loader, test_label_loader): 
        data = [d.to(device) for d in data]
        seg_data = [d.to(device) for d in seg_data]
        x, y, x_seg, y_seg = data[0], data[1], seg_data[0], seg_data[1]
        
        start = time.time()
        flow = model((x, y))
        #print("Flow shape:", flow.shape)
        #print("Flow shape dims:", flow.shape)
        #print("Flow dtype:", flow.dtype)
        #print("x shape:", x.shape)
        #print("y shape:", y.shape)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.time() - start
        
        # Save deformation field (flow)
        i+=1
        flow_np = flow[1].cpu().numpy()[0]  # shape: (3, H, W, D)
        np.save(f"{logs_folder}/flows/flow_{i}.npy", flow_np)

        # Save image pair index if needed
        np.save(f"{logs_folder}/pairs/moving_{i}.npy", x.cpu().numpy()[0])
        np.save(f"{logs_folder}/pairs/fixed_{i}.npy", y.cpu().numpy()[0])
        
        warped_list = []
        for i in range(x_seg.shape[2]):  # 22 slices along channel dim
            seg_i = x_seg[:, :, i]  # shape: [1, 1, D, H, W]
            warped_i = reg_model(seg_i.to(device).float(), flow[1].to(device))  # returns [1, 1, D, H, W]
            warped_list.append(warped_i)

        warped_label_oh = torch.cat(warped_list, dim=1)  # shape: [1, 1, 22, D, H, W]

        # Warp one-hot segmentation map
        # new_locs = flow_to_grid(flow) # (B, D, H, W, 3) -> (B, 3, D, H, W)
        # warped_label_oh = reg_model(x_seg.to(device).long(), flow[1].to(device))
        
        # Convert warped to hard label
        warped_label_oh = warped_label_oh.squeeze(1)             # [1, 22, D, H, W]
        warped_label = torch.argmax(warped_label_oh, dim=1)      # [1, D, H, W]

        # Compute Jacobian ratio
        jac_ratio = compute_non_positive_jacobian_ratio(flow[1])
        all_jac.append(jac_ratio)

        # Loop over batch to compute DICE & HD95
        print(f"batch: {warped_label.size(0)}")
        for i in range(warped_label.size(0)):
            pred = warped_label[i].cpu().numpy()
            gt = torch.argmax(y_seg[i, 0], dim=0).cpu().numpy()

            print(f"pred shape: {pred.shape}")
            print(f"gt shape: {gt.shape}")
            dice = compute_label_wise_dice(pred, gt, label_list)
            hd_95 = compute_label_wise_95hd(pred, gt, label_list)

            all_dice.append(dice)
            all_hd95.append(hd_95)

        # warped_np = warped_label.argmax(1).squeeze().detach().cpu().numpy()
        # y_np = y_seg.squeeze().detach().cpu().numpy()
        # dice_results = compute_dice_per_organ(warped_np, y_np, labels, patient_id)
        # all_dice_per_organ.extend(dice_results)
        
        # all_dice.append(result['DICE'])
        # all_hd95.append(result['HD95'])
        # all_jac.append(jac_ratio)
        all_time.append(elapsed / x.shape[0]) 

        #print(f"Dice: {dice:.4f}, HD95: {hd:.2f}, Jacobian ≤ 0: {jac_ratio:.4f}, Inference Time: {elapsed:.3f}s")

        print("Done")

# Final metrics
final_dice = np.nanmean(np.array(all_dice))
final_dice_std = np.nanstd(np.array(all_dice))
final_dice_organ = np.nanmean(np.array(all_dice), axis=0)
final_dice_organ_std = np.nanstd(np.array(all_dice), axis=0)

flat_hd95 = [v for sublist in all_hd95 for v in sublist if not np.isnan(v)]

if len(flat_hd95) == 0:
    print("Warning: No valid HD95 values found.")
    final_hd95 = np.nan
    final_hd95_std = np.nan
else:
    final_hd95 = np.mean(flat_hd95)
    final_hd95_std = np.std(flat_hd95)

#final_hd95 = np.nanmean(np.array(all_hd95))
#final_hd95_std = np.nanstd(np.array(all_hd95))

final_jac = np.nanmean(np.array(all_jac))
final_jac_std = np.nanstd(np.array(all_jac))

final_time = sum(all_time) / len(all_time)

# df = pd.DataFrame(all_dice_results)
# mean_dice_per_organ = df.groupby('organ')['dice'].mean()
# std_dice_per_organ = df.groupby('organ')['dice'].std()

print(f"\n--- Final Evaluation Metrics ---")
print(f"Dice Score       : {final_dice:.4f} ± {final_dice_std:.4f}")
print(f"Dice by organ    : {final_dice_organ} +/- {final_dice_organ_std}")
print(f"HD95             : {final_hd95:.2f} ± {final_hd95_std:.2f}")
print(f"Jacobian ≤ 0     : {final_jac:.4%} ± {final_jac_std:.4%}")
print(f"Inference Time   : {final_time:.4f} sec/sample")
print(f"Number of parameters: {n_params / 1e6:.2f}M")
# print(f"Dice per organ   : {mean_dice_per_organ:.4f} +/- {std_dice_per_organ:.4f}")

csv_path = '{logs_folder}/voxelmorph_results.csv'
save_metrics_csv({
    'Dice Score': final_dice,
    'Dice Score Std': final_dice_std,
    'Dice by organ': final_dice_organ,
    'Dice by organ Std': final_dice_organ_std,
    'HD95': final_hd95,  
    'HD95 Std': final_hd95_std,   
    'Jacobian ≤ 0': final_jac,
    'Jacobian ≤ 0 Std': final_jac_std,
    'Inference Time (sec/sample)': final_time,
    'Number of parameters (M)': n_params / 1e6
    }, csv_path)

'''
# Load your existing CSV summary (if it exists)
try:
    df_summary = pd.read_csv(csv_path)
except FileNotFoundError:
    df_summary = pd.DataFrame()  # Or create a new one if not exists

# Calculate mean dice per organ from your per-case results
mean_dice_per_organ = df.groupby('organ')['dice'].mean()

# Add these as new columns to the summary dataframe
# e.g. columns: 'Dice_liver', 'Dice_spleen', etc.
for organ, mean_dice in mean_dice_per_organ.items():
    col_name = f'Dice_{organ}'
    df_summary.at[0, col_name] = mean_dice  # put in first row or append new row if needed

# If your df_summary is empty, also add the other overall metrics as first row (optional)
if df_summary.empty:
    df_summary = pd.DataFrame({
        'Dice Score': [final_dice],
        'HD95': [final_hd95],
        'Jacobian ≤ 0': [final_jac],
        'Inference Time (sec/sample)': [final_time],
        'Number of parameters (M)': [n_params / 1e6],
    })
    # Then add organ dice as above:
    for organ, mean_dice in mean_dice_per_organ.items():
        col_name = f'Dice_{organ}'
        df_summary.at[0, col_name] = mean_dice

# Save the updated CSV
df_summary.to_csv(csv_path, index=False)
'''

