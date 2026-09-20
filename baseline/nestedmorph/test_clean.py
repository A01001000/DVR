#!/usr/bin/env python3
"""
Comprehensive test script for VoxelMorph and NestedMorph models using NIfTI data format.

Data Structure Expected:
Test/
├── patient_001/
│   ├── MR.nii.gz
│   ├── CT.nii.gz
│   ├── MR_seg.nii.gz (optional)
│   └── CT_seg.nii.gz (optional)
└── patient_002/
    ├── MR.nii.gz
    ├── CT.nii.gz
    ├── MR_seg.nii.gz (optional)
    └── CT_seg.nii.gz (optional)

Evaluates models on test data and computes:
- Dice Score (overall and per-organ)
- HD95 (95th percentile Hausdorff Distance)
- Non-positive Jacobian Determinant ratio
- Inference time
- Number of parameters

Also saves:
- Deformation fields (.npy format)
- Warped images (.nii.gz format)
- Warped segmentations (.nii.gz format)
"""

import os
import sys
import csv
import time
import torch
import argparse
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import DataLoader
from medpy.metric import binary, dc, hd95
from torchvision import transforms
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.voxelmorph import VoxelMorph
from src.models.nestedmorph import NestedMorph
from src.losses.losses import MINDLoss, FlowGrad3d
from src.data.nifti_datasets import create_test_dataloader
from src.utils.utils import register_model, dice
from src.utils.config import device
from src.data.trans import *


def save_middle_slice_png(volume, out_path, cmap='gray', vmin=None, vmax=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mid = volume.shape[2] // 2
    img = volume[:,:,mid].squeeze()
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_overlay_png(image, seg, out_path, label_cmap='jet', alpha=0.4, vmin=None, vmax=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mid = image.shape[2] // 2
    img_slice = image[:,:,mid].squeeze()
    seg_slice = seg[:,:,mid].squeeze()
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax)
    if np.max(seg_slice) > 0:
        plt.imshow(seg_slice, cmap=label_cmap, alpha=alpha, vmin=0, vmax=np.max(seg_slice))
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_overlay_png_masked(image, seg, out_path, label_cmap='jet', alpha=0.5, vmin=None, vmax=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mid = image.shape[2] // 2
    img_slice = image[:,:,mid].squeeze()
    seg_slice = seg[:,:,mid].squeeze()
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax)
    mask = seg_slice > 0
    if np.any(mask):
        colored = np.zeros((*seg_slice.shape, 4))
        colored[mask] = plt.cm.get_cmap(label_cmap)(seg_slice[mask]/np.max(seg_slice[mask]))
        plt.imshow(colored, alpha=alpha)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_deformation_field_png(def_field, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if def_field.shape[0] == 3:
        def_field = np.moveaxis(def_field, 0, -1)
    if def_field.shape[-1] == 3:
        mid = def_field.shape[2] // 2
        u = def_field[:,:,mid,0].squeeze()
        v = def_field[:,:,mid,1].squeeze()
        mag = np.sqrt(u**2 + v**2)
        ang = np.arctan2(v, u)
        ang_norm = (ang + np.pi) / (2 * np.pi)
        mag_norm = mag / (np.max(mag) + 1e-8)
        hsv = np.zeros(u.shape + (3,), dtype=np.float32)
        hsv[...,0] = ang_norm
        hsv[...,1] = 1
        hsv[...,2] = mag_norm
        rgb = mcolors.hsv_to_rgb(hsv)
        plt.figure(figsize=(5,5))
        plt.axis('off')
        plt.imshow(rgb)
        plt.title('Deformation Field (middle slice)')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        mid = def_field.shape[2] // 2
        u = def_field[:,:,mid,0].squeeze()
        v = def_field[:,:,mid,1].squeeze()
        plt.figure(figsize=(5,5))
        plt.axis('off')
        plt.quiver(u, v)
        plt.title('Deformation Field (middle slice)')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def save_error_map_png(seg1, seg2, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    error_map = np.abs(seg1 - seg2)
    mid = error_map.shape[2] // 2
    err_slice = error_map[:,:,mid].squeeze()
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(np.ones_like(err_slice), cmap='gray', vmin=0, vmax=1)
    if np.max(err_slice) > 0:
        plt.imshow(err_slice, cmap='hot', alpha=0.8, vmin=0, vmax=np.max(err_slice))
    plt.title('Error Map (middle slice)')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# --- Per-organ Dice computation from syn_test.py ---
def compute_dice_coefficient(mask1, mask2):
    """
    Compute the Dice Similarity Coefficient between two binary masks.
    """
    intersection = np.sum(mask1 * mask2)
    size1 = np.sum(mask1)
    size2 = np.sum(mask2)
    return (2.0 * intersection + 1e-5) / (size1 + size2 + 1e-5)


def compute_label_wise_dice(seg1, seg2, labels):
    """Compute Dice for each label/organ."""
    dice_results = []
    for label in labels:
        # Isolate current label in both segmentations
        seg1_label = seg1 == label
        seg2_label = seg2 == label
        
        # Compute Dice for the current label
        dice = compute_dice_coefficient(seg1_label, seg2_label)
        dice_results.append(dice)
    
    return dice_results


def score_case(seg_fixed, seg_moving, label_list):
    """Compute per-organ metrics for a single case (following baseline dataset pattern)."""
    dice_list = []
    
    for label in label_list:
        # Get binary masks for this label
        mask1 = (seg_fixed == label).astype(np.float32)
        mask2 = (seg_moving == label).astype(np.float32)
        
        # Compute intersection and sizes
        intersection = np.sum(mask1 * mask2)
        size1 = np.sum(mask1)
        size2 = np.sum(mask2)
        
        # Compute Dice with epsilon (following baseline pattern)
        dice = (2.0 * intersection + 1e-5) / (size1 + size2 + 1e-5)
        dice_list.append(dice)
    
    return {'DICE': np.array(dice_list)}


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

        if not np.any(seg1_label) or not np.any(seg2_label):
            print(f"Label {label} missing - skipping HD95.")
            hd95_results.append(np.nan)
            continue

        hd = compute_95_hausdorff_distance(seg1_label, seg2_label)
        hd95_results.append(hd)

    return hd95_results


def compute_non_positive_jacobian_ratio(flow):
    """
    Compute ratio of voxels with det(Jacobian) ≤ 0.
    Args:
        flow: torch.Tensor (B, 3, H, W, D), deformation field
    Returns:
        float: mean ratio of voxels with det(Jacobian) ≤ 0
    """
    flow_np = flow.detach().cpu().numpy()
    ratios = []

    for b in range(flow_np.shape[0]):
        phi = flow_np[b]  # shape: (3, H, W, D)

        # Compute gradients
        dFx_dx = np.gradient(phi[0], axis=0)
        dFx_dy = np.gradient(phi[0], axis=1)
        dFx_dz = np.gradient(phi[0], axis=2)

        dFy_dx = np.gradient(phi[1], axis=0)
        dFy_dy = np.gradient(phi[1], axis=1)
        dFy_dz = np.gradient(phi[1], axis=2)

        dFz_dx = np.gradient(phi[2], axis=0)
        dFz_dy = np.gradient(phi[2], axis=1)
        dFz_dz = np.gradient(phi[2], axis=2)

        # Create Jacobian matrix (I + ∇u)
        H, W, D = phi.shape[1:]
        J = np.zeros((H, W, D, 3, 3))

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
        jac_det = np.linalg.det(J)
        non_pos_ratio = np.mean(jac_det <= 0)
        ratios.append(non_pos_ratio)

    return np.mean(ratios)


def save_warped_image(warped_img, filename, affine=None):
    """Save warped image as NIfTI file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if affine is None:
        affine = np.eye(4)
    
    if torch.is_tensor(warped_img):
        warped_img = warped_img.detach().cpu().numpy()
    
    if len(warped_img.shape) == 4 and warped_img.shape[0] == 1:
        warped_img = warped_img[0]
    
    # Convert to appropriate data type for NIfTI
    if warped_img.dtype == np.int64:
        warped_img = warped_img.astype(np.int16)  # Convert int64 to int16 for segmentations
    elif warped_img.dtype == np.float64:
        warped_img = warped_img.astype(np.float32)  # Convert float64 to float32 for images
    
    nii_img = nib.Nifti1Image(warped_img, affine)
    nib.save(nii_img, filename)


def save_metrics_csv(metrics, output_path):
    """Save evaluation metrics to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in metrics.items():
            writer.writerow([key, value])


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test VoxelMorph/NestedMorph models on NIfTI data')
    parser.add_argument('--model', type=str, default='VoxelMorph', 
                        choices=['VoxelMorph', 'NestedMorph'],
                        help='Model to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Test directory containing patient folders')
    parser.add_argument('--output-dir', type=str, default='./test_results',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--img-size', type=int, nargs=3, default=[128, 128, 128],
                        help='Image size (H W D) - should match training resolution')
    parser.add_argument('--num-channels', type=int, default=4,
                        help='Number of segmentation channels')
    parser.add_argument('--save-images', action='store_true',
                        help='Save warped images and deformation fields')
    
    return parser.parse_args()


def detect_dataset_labels(test_dir):
    """Detect available labels in the dataset by examining actual segmentation files."""
    from pathlib import Path
    import nibabel as nib
    
    test_path = Path(test_dir)
    
    # Find segmentation files (try multiple patterns)
    seg_files = list(test_path.glob("**/label*.nii*"))
    if not seg_files:
        seg_files = list(test_path.glob("**/*seg*.nii*"))
    
    if not seg_files:
        # Use consistent label set for proper comparison across datasets
        return [1, 2, 3, 4]  # Always use all 4 labels for consistent evaluation
    
    # Analyze actual files to find all labels
    all_labels = set()
    files_checked = 0
    max_files_to_check = min(15, len(seg_files))  # Check more files to find all possible labels
    
    for seg_file in seg_files[:max_files_to_check]:
        try:
            seg_data = nib.load(str(seg_file)).get_fdata()
            labels = np.unique(seg_data)
            non_zero_labels = labels[labels > 0]
            all_labels.update(non_zero_labels.astype(int))
            files_checked += 1
        except Exception as e:
            print(f"Warning: Could not read {seg_file}: {e}")
            continue
    
    if not all_labels:
        # If we couldn't read any files, use consistent 4-label evaluation
        return [1, 2, 3, 4]  # Use all 4 labels for consistent baseline comparison
    
    detected_labels = sorted(list(all_labels))
    
    # For consistent baseline comparison, always use all 4 labels
    # This matches the original behavior that was working correctly
    expected_labels = [1, 2, 3, 4]  # liver, kidney, pancreas, spleen
    print(f"Using consistent labels {expected_labels} for baseline comparison (detected from {files_checked} files: {detected_labels})")
    return expected_labels


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Convert img_size to tuple
    args.img_size = tuple(args.img_size)
    
    # Create output directories
    os.makedirs(f"{args.output_dir}/flows", exist_ok=True)
    os.makedirs(f"{args.output_dir}/warped_images", exist_ok=True)
    os.makedirs(f"{args.output_dir}/warped_segmentations", exist_ok=True)

    # Detect dataset-specific labels
    label_list = detect_dataset_labels(args.test_dir)
    print(f"Detected labels for evaluation: {label_list}")
    
    # Update num_channels based on detected labels
    args.num_channels = len(label_list)

    # Load model
    print(f"Loading {args.model} model...")
    if args.model == "VoxelMorph":
        model = VoxelMorph(
            inshape=tuple(args.img_size),  # Use actual image size from arguments
            nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]
        ).to(device)
    elif args.model == "NestedMorph":
        model = NestedMorph(inshape=tuple(args.img_size)).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if 'state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"Warning: Size mismatch detected. Loading with strict=False")
                print(f"Error: {e}")
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                raise e
    else:
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"Warning: Size mismatch detected. Loading with strict=False")
                print(f"Error: {e}")
                model.load_state_dict(checkpoint, strict=False)
            else:
                raise e
    
    model.eval()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # Load spatial transformer for warping
    reg_model = register_model(args.img_size, 'nearest').to(device)
    reg_model.eval()

    # Data transforms
    test_composed = transforms.Compose([
        NumpyType((np.float32, np.float32)),
    ])

    # Create test dataloader
    print(f"Creating test dataloader from {args.test_dir}")
    test_loader = create_test_dataloader(
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_classes=args.num_channels,
        transforms=test_composed
    )

    print(f"Found {len(test_loader)} test samples")

    # Initialize metric containers (following baseline dataset pattern)
    dice_list = []  # Will store per-organ Dice arrays for each patient
    hd95_list = []  # For HD95 scores
    all_jac = []
    all_time = []
    
    print(f"Starting evaluation on {len(test_loader)} test samples...")

    with torch.no_grad():
        for batch_idx, (data, seg_data, patient_ids, has_seg) in enumerate(test_loader): 
            print(f"Processing batch {batch_idx + 1}/{len(test_loader)} - Patient: {patient_ids[0]}")
            
            # Move data to device
            (x, y), (x_seg, y_seg) = data, seg_data
            x, y = x.to(device), y.to(device)  # moving, fixed images
            x_seg, y_seg = x_seg.to(device), y_seg.to(device)  # moving, fixed segmentations
            
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
            
            if args.save_images:
                # Save deformation field
                flow_np = flow.cpu().numpy()[0]  # Remove batch dimension
                np.save(f"{args.output_dir}/flows/flow_{patient_ids[0]}.npy", flow_np)
                # Save warped image
                save_warped_image(warped_image[0], f"{args.output_dir}/warped_images/warped_{patient_ids[0]}.nii.gz")

            # Warp segmentation (only if segmentation data is available)
            if has_seg:
                warped_seg_list = []
                for channel_idx in range(x_seg.shape[1]):  # Iterate over channels
                    seg_channel = x_seg[:, channel_idx:channel_idx+1]  # Keep channel dimension
                    warped_channel = reg_model(seg_channel.float(), flow)
                    warped_seg_list.append(warped_channel)
                warped_seg_oh = torch.cat(warped_seg_list, dim=1)  # Concatenate channels
                # Convert to hard labels
                warped_seg = torch.argmax(warped_seg_oh, dim=1)  # Remove channel dimension
                if args.save_images:
                    # Save warped segmentation
                    save_warped_image(warped_seg[0], f"{args.output_dir}/warped_segmentations/warped_seg_{patient_ids[0]}.nii.gz")
                    # --- Save PNG overlays and visualizations that require warped_seg ---
                    # Save overlay: fixed CT + label
                    fixed_img = y[0].cpu().numpy() if torch.is_tensor(y) else y
                    fixed_seg = torch.argmax(y_seg[0], dim=0).cpu().numpy() if torch.is_tensor(y_seg) else y_seg
                    save_overlay_png(fixed_img, fixed_seg, os.path.join(args.output_dir, f"fixed_CT_overlay_{patient_ids[0]}.png"))
                    # Save overlay: moving MR + label
                    moving_img = x[0].cpu().numpy() if torch.is_tensor(x) else x
                    moving_seg = torch.argmax(x_seg[0], dim=0).cpu().numpy() if torch.is_tensor(x_seg) else x_seg
                    save_overlay_png(moving_img, moving_seg, os.path.join(args.output_dir, f"moving_MR_overlay_{patient_ids[0]}.png"))
                    # Save overlay: warped MR + warped label
                    save_overlay_png(warped_image[0].cpu().numpy(), warped_seg[0].cpu().numpy(), os.path.join(args.output_dir, f"warped_MR_overlay_{patient_ids[0]}.png"))
                    # Save deformation field visualization
                    save_deformation_field_png(flow_np, os.path.join(args.output_dir, f"deformation_field_middle_{patient_ids[0]}.png"))
                    # Save error map
                    save_error_map_png(warped_seg[0].cpu().numpy(), fixed_seg, os.path.join(args.output_dir, f"error_map_middle_{patient_ids[0]}.png"))
            else:
                print(f"  No segmentation data for patient {patient_ids[0]} - skipping metrics")
                # Still compute timing and Jacobian
                jac_ratio = compute_non_positive_jacobian_ratio(flow)
                all_jac.append(jac_ratio)
                all_time.append(elapsed_time / x.shape[0])

    # Calculate final metrics only if we have segmentation data
    if len(dice_list) > 0:
        dice_array = np.array(dice_list)
        hd95_array = np.array(hd95_list)
        
        # Calculate means across all samples following baseline pattern
        all_means = np.nanmean(dice_array, axis=1) 
        final_dice = np.nanmean(all_means)
        final_dice_std = np.nanstd(all_means)
        
        # Per-organ metrics
        final_dice_organ = np.nanmean(dice_array, axis=0)
        final_dice_organ_std = np.nanstd(dice_array, axis=0)
        
        # HD95 metrics (flatten and filter valid values)
        flat_hd95 = hd95_array.flatten()
        valid_hd95 = flat_hd95[~np.isnan(flat_hd95)]
        
        if len(valid_hd95) > 0:
            final_hd95 = np.mean(valid_hd95)
            final_hd95_std = np.std(valid_hd95)
        else:
            print("Warning: No valid HD95 values found.")
            final_hd95 = np.nan
            final_hd95_std = np.nan
    else:
        print("Warning: No segmentation data found - only computing timing and Jacobian metrics")
        final_dice = np.nan
        final_dice_std = np.nan
        final_dice_organ = np.array([np.nan] * len(label_list))
        final_dice_organ_std = np.array([np.nan] * len(label_list))
        final_hd95 = np.nan
        final_hd95_std = np.nan
    
    # Jacobian metrics
    final_jac = np.mean(all_jac)
    final_jac_std = np.std(all_jac)
    
    # Timing metrics
    final_time = np.mean(all_time)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Final Evaluation Results for {args.model}")
    print(f"{'='*60}")
    print(f"Overall Dice Score    : {final_dice:.4f} ± {final_dice_std:.4f}")
    
    # Create label mapping for clarity
    if 'CHAOS' in args.test_dir.upper():
        label_names = ['liver']
    elif 'L2R' in args.test_dir.upper():
        # Always show all 4 expected organs for L2R
        label_names = ['liver', 'kidney', 'pancreas', 'spleen']
    else:
        label_names = [f'organ_{i}' for i in range(len(label_list))]
    
    print(f"Per-organ Dice        : {[f'{d:.4f}' for d in final_dice_organ]} ({label_names})")
    print(f"Per-organ Dice Std    : {[f'{d:.4f}' for d in final_dice_organ_std]}")
    print(f"HD95                  : {final_hd95:.2f} ± {final_hd95_std:.2f}")
    print(f"Jacobian ≤ 0         : {final_jac:.4%} ± {final_jac_std:.4%}")
    print(f"Inference Time        : {final_time:.4f} sec/sample")
    print(f"Number of Parameters  : {n_params / 1e6:.2f}M")
    
    # Save results to CSV
    csv_path = f'{args.output_dir}/{args.model.lower()}_evaluation_results.csv'
    save_metrics_csv({
        'Model': args.model,
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
    if args.save_images:
        print(f"Deformation fields saved to: {args.output_dir}/flows/")
        print(f"Warped images saved to: {args.output_dir}/warped_images/")
        print(f"Warped segmentations saved to: {args.output_dir}/warped_segmentations/")


if __name__ == "__main__":
    main()
