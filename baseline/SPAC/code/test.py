import torch
import torch.nn.functional as F
import random
import numpy as np
import copy
import cv2
import time
import argparse
import nibabel as nib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import csv

# Fix matplotlib cache directory issue
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.makedirs('/tmp', exist_ok=True)  # Ensure /tmp exists for matplotlib cache

import utils
import data_util
import data_util.brain
import data_util.liver
import data_util.custom
from config import Config as cfg
from brain import SPAC
from env import Env
from summary import Summary
from networks import *

# For HD95 calculation
try:
    from medpy.metric.binary import hd95
    MEDPY_AVAILABLE = True
except ImportError:
    print("Warning: medpy not available. HD95 calculation will be skipped.")
    MEDPY_AVAILABLE = False


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
    dice_coefficient = compute_label_wise_dice(seg_fixed, seg_moving, label_list)
    dice_coefficient = np.array(dice_coefficient)
    
    return {'DICE': dice_coefficient}


def detect_dataset_labels(data_path_or_name):
    """Detect available labels based on dataset name or path."""
    dataset_name = data_path_or_name.upper() if isinstance(data_path_or_name, str) else str(data_path_or_name).upper()
    
    if 'CHAOS' in dataset_name:
        return [1]  # CHAOS only has liver (label 1)
    elif 'L2R' in dataset_name:
        return [1, 2, 3, 4]  # L2R has liver, kidney, pancreas, spleen
    else:
        return [1, 2, 3, 4]  # Default fallback


# os.environ['CUDA_VISIBLE_DEVICES'] = pa.GPU_ID

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(cfg.GPU_ID)
else:
    device = torch.device('cpu')

# device = torch.device('cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True # cpu\gpu 结果一致

def parse_args():
    parser = argparse.ArgumentParser(description='Test SPAC Model')
    parser.add_argument('--data_type', type=str, default=cfg.DATA_TYPE, help='Dataset type')
    parser.add_argument('--test_data', type=str, default=cfg.DATA_TYPE, help='Dataset type')
    parser.add_argument('--save_warped', action='store_true', help='Save warped images and deformation fields')
    args = parser.parse_args()
    return args

def update_config(updates):
    """
    Updates attributes of the imported 'cfg' object directly in memory.

    Args:
        updates (dict): A dictionary with keys matching the attributes
                        to update and their new values.
    """
    for key, value in updates.items():
        # Check if the attribute exists before trying to set it
        if hasattr(cfg, key):
            setattr(cfg, key, value)
            print(f"Updated '{key}' to: {value}")
        else:
            print(f"Warning: Attribute '{key}' not found in config. Ignoring.")

def count_parameters(model):
    """Count the number of parameters in a model (both trainable and non-trainable)."""
    return sum(p.numel() for p in model.parameters())

def calculate_negative_jacobian_percentage(flow):
    """Calculate the percentage of voxels with negative Jacobian determinant."""
    # flow shape: [H, W, D, 3]
    jac_det = utils.jacobian_determinant(flow)
    negative_jac = jac_det <= 0
    return (negative_jac.sum() / negative_jac.size) * 100

def save_nifti(data, filepath, affine=None):
    """Save numpy array as NIfTI file."""
    if affine is None:
        affine = np.eye(4)
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, filepath)

def save_middle_slice_png(volume, out_path, cmap='gray', vmin=None, vmax=None):
    mid = volume.shape[2] // 2
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(volume[:,:,mid], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_overlay_png(image, seg, out_path, label_cmap='jet', alpha=0.4, vmin=None, vmax=None):
    mid = image.shape[2] // 2
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(image[:,:,mid], cmap='gray', vmin=vmin, vmax=vmax)
    seg_slice = seg[:,:,mid]
    if np.max(seg_slice) > 0:
        plt.imshow(seg_slice, cmap=label_cmap, alpha=alpha, vmin=0, vmax=np.max(seg_slice))
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_deformation_field_png(def_field, out_path):
    if def_field.shape[0] == 3:
        def_field = np.moveaxis(def_field, 0, -1)
    if def_field.shape[-1] == 3:
        mid = def_field.shape[2] // 2
        u = def_field[:,:,mid,0]
        v = def_field[:,:,mid,1]
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
        u = def_field[:,:,mid,0]
        v = def_field[:,:,mid,1]
        plt.figure(figsize=(5,5))
        plt.axis('off')
        plt.quiver(u, v)
        plt.title('Deformation Field (middle slice)')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def save_error_map_png(seg1, seg2, out_path):
    error_map = np.abs(seg1 - seg2)
    mid = error_map.shape[2] // 2
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(np.ones_like(error_map[:,:,mid]), cmap='gray', vmin=0, vmax=1)
    if np.max(error_map[:,:,mid]) > 0:
        plt.imshow(error_map[:,:,mid], cmap='hot', alpha=0.8, vmin=0, vmax=np.max(error_map[:,:,mid]))
    plt.title('Error Map (middle slice)')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_dice_csv(dices, label_list, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample'] + [f'Organ_{l}_Dice' for l in label_list])
        for i, dice_row in enumerate(dices):
            writer.writerow([i] + list(dice_row))

if __name__ == "__main__":
    args = parse_args()
    updates = {
        "DATA_TYPE": args.data_type, # liver, brain, mrxfdg
        "TEST_DATA": args.test_data
    }
    update_config(updates)
    idx = 100
    setup_seed(cfg.SEED)
    utils.remkdir(cfg.TEST_PATH)

    #######################################
    # stn = SpatialTransformer(device).to(device)
    stn = SpatialTransformer(cfg.HEIGHT).to(device)
    seg_stn = SpatialTransformer(cfg.HEIGHT, mode='nearest').to(device)

    Dataset = eval('data_util.{}.Dataset'.format(cfg.IMAGE_TYPE if cfg.IMAGE_TYPE else 'custom'))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'datasets', f'{cfg.DATA_TYPE}.json')
    dataset = Dataset(split_path=dataset_path, paired=True, affine=False)
    
    # Use the first available scheme key (typically "1") 
    # The scheme defines train/validation split ratios, not subset names
    # Explicitly load the validation set for testing
    print(f"Dataset has {len(dataset.subset.get('train', []))} train samples and {len(dataset.subset.get('validation', []))} validation samples.")

    # We will find a scheme that contains the 'validation' subset
    validation_scheme_key = None
    for key, scheme_subsets in dataset.schemes.items():
        if 'train' in scheme_subsets:
            validation_scheme_key = key
            break

    if validation_scheme_key is not None:
        print(f"Found train data in scheme: {validation_scheme_key}. Creating test loader.")
        test_loader = dataset.generator(validation_scheme_key, subset_name='train', batch_size=1, loop=False)
    else:
        raise ValueError("Could not find a 'train' subset in any of the schemes in your JSON file.")

    brain = SPAC(stn, device)
    
    # Check if model files exist before loading
    model_files = {
        'actor': cfg.ACTOR_MODEL_RL_EVAL,  # Use evaluation model (loads actor_*.ckpt for self.actor which is Decoder)
        'planner': cfg.PLANNER_MODEL_RL_EVAL if hasattr(cfg, 'PLANNER_MODEL_RL_EVAL') else None,  # Use evaluation model (loads planner_*.ckpt for self.planner which is Encoder)
        'critic1': cfg.CRITIC1_MODEL_EVAL if hasattr(cfg, 'CRITIC1_MODEL_EVAL') else None,  # Use evaluation model
        'critic2': cfg.CRITIC2_MODEL_EVAL if hasattr(cfg, 'CRITIC2_MODEL_EVAL') else None  # Use evaluation model
    }
    
    import os
    for model_name, model_path in model_files.items():
        if model_path and not os.path.exists(model_path):
            print(f"Warning: {model_name} model file not found: {model_path}")
            print(f"Please make sure you have trained the model or provide the correct model path.")
            # For now, skip loading missing models to allow testing of basic functionality
            continue
        elif model_path:
            print(f"Loading {model_name} from: {model_path}")
            brain.load_model(model_name, model_path)

    brain.eval(brain.actor)
    brain.eval(brain.critic1)
    brain.eval(brain.critic2)
    brain.eval(brain.planner)

    # Count total parameters with debug info
    actor_params = count_parameters(brain.actor)
    planner_params = count_parameters(brain.planner)
    critic1_params = count_parameters(brain.critic1)
    critic2_params = count_parameters(brain.critic2)
    
    print(f"Parameter counts:")
    print(f"  Actor: {actor_params:,}")
    print(f"  Planner: {planner_params:,}")
    print(f"  Critic1: {critic1_params:,}")
    print(f"  Critic2: {critic2_params:,}")
    
    total_params = actor_params + planner_params + critic1_params + critic2_params
    print(f"Total number of parameters: {total_params:,}")

    # Detect dataset labels for per-organ evaluation
    label_list = detect_dataset_labels(cfg.DATA_TYPE)
    print(f"Detected labels for evaluation: {label_list}")

    # Initialize metrics storage
    dices = []  # Will store per-organ Dice arrays for each patient
    hd95s = []
    neg_jac_percentages = []
    inference_times = []
    
    # Count samples manually since test_loader is a generator
    test_samples = []
    for sample in test_loader:
        test_samples.append(sample)
    
    print(f"Testing on {len(test_samples)} samples")
    
    for i, data in enumerate(test_samples):
        # Extract data from dictionary format
        if isinstance(data, dict):
            # Data is in dictionary format from generator
            fixed = data['fixed'][0, 0]  # Remove batch and channel dims
            moving = data['moving'][0, 0]
            
            # Check if segmentation data exists
            has_segmentation = 'fixed_seg' in data and 'moving_seg' in data
            if has_segmentation:
                fixed_seg = data['fixed_seg'][0, 0]
                moving_seg = data['moving_seg'][0, 0]
            else:
                fixed_seg = moving_seg = None
        else:
            # Handle tuple format (legacy)
            if len(data) == 4:
                # Data with segmentation
                fixed, moving, fixed_seg, moving_seg = data
                has_segmentation = True
            else:
                # Data without segmentation
                fixed, moving = data
                has_segmentation = False
                fixed_seg = moving_seg = None
                
        if has_segmentation:
            # Ensure they are numpy arrays before comparing
            fixed_seg_np_check = fixed_seg if isinstance(fixed_seg, np.ndarray) else fixed_seg.cpu().numpy()
            moving_seg_np_check = moving_seg if isinstance(moving_seg, np.ndarray) else moving_seg.cpu().numpy()
            if np.array_equal(fixed_seg_np_check, moving_seg_np_check):
                print(f"    🚨 WARNING for sample {i}: Initial fixed and moving segmentations are identical!")
                
        if i % 1 == 0:
            # Ensure fixed and moving are numpy arrays for image saving
            fixed_np = fixed if isinstance(fixed, np.ndarray) else fixed.cpu().numpy()
            moving_np = moving if isinstance(moving, np.ndarray) else moving.cpu().numpy()
            cv2.imwrite('{}/{}-fixed.bmp'.format(cfg.TEST_PATH, i), utils.numpy_im(fixed_np)[:, :, idx])
            cv2.imwrite('{}/{}-moving.bmp'.format(cfg.TEST_PATH, i), utils.numpy_im(moving_np)[:, :, idx])

        # Convert to torch tensors and move to device
        if not isinstance(fixed, torch.Tensor):
            fixed = torch.from_numpy(fixed).to(device).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        else:
            fixed = fixed.to(device)
            
        if not isinstance(moving, torch.Tensor):
            moving = torch.from_numpy(moving).to(device).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        else:
            moving = moving.to(device)
        
        if has_segmentation:
            if not isinstance(fixed_seg, torch.Tensor):
                fixed_seg_torch = torch.from_numpy(fixed_seg).to(device).unsqueeze(0).unsqueeze(0)
            else:
                fixed_seg_torch = fixed_seg.to(device)
                
            if not isinstance(moving_seg, torch.Tensor):
                moving_seg_torch = torch.from_numpy(moving_seg).to(device).unsqueeze(0).unsqueeze(0)
            else:
                moving_seg_torch = moving_seg.to(device)

        moved = copy.deepcopy(moving)

        pred = None
        step = 0
        tic = time.time()

        # Registration process
        print(f"    Starting registration for sample {i}...")
        while step < 20:
            state = torch.cat([fixed, moved], dim=1)
            # Ensure state is 5D: [B, C, D, H, W]
            while state.ndim < 5:
                state = state.unsqueeze(0)
            latent, flow = brain.choose_action(state, test=True)
            
            # Debug: Check flow statistics
            if step == 0:  # Only print for first step to avoid spam
                flow_stats = {
                    'min': flow.min().item(),
                    'max': flow.max().item(),
                    'mean': flow.mean().item(),
                    'std': flow.std().item(),
                    'shape': flow.shape
                }
                print(f"    Flow stats (step {step}): min={flow_stats['min']:.6f}, max={flow_stats['max']:.6f}, mean={flow_stats['mean']:.6f}, std={flow_stats['std']:.6f}")
                
            pred = flow if pred is None else stn(pred, flow) + flow
            moved = stn(moving, pred)
            step += 1

        toc = time.time()
        inference_time = toc - tic
        inference_times.append(inference_time)

        # Apply final transformation
        warped_im = utils.numpy_im(stn(moving, pred), device=device)
        
        # Calculate negative Jacobian percentage
        flow_numpy = utils.numpy(pred.squeeze(), device=device)
        flow_for_jac = np.transpose(flow_numpy, (1, 2, 3, 0))  # [H, W, D, 3]
        neg_jac_percent = calculate_negative_jacobian_percentage(flow_for_jac)
        neg_jac_percentages.append(neg_jac_percent)
        
        # Debug: Check final deformation field statistics
        pred_stats = {
            'min': pred.min().item(),
            'max': pred.max().item(),
            'mean': pred.mean().item(),
            'std': pred.std().item()
        }
        print(f"    Final deformation stats: min={pred_stats['min']:.6f}, max={pred_stats['max']:.6f}, mean={pred_stats['mean']:.6f}, std={pred_stats['std']:.6f}")
        
        # Check if deformation is essentially zero (identity transformation)
        if abs(pred_stats['mean']) < 1e-6 and pred_stats['std'] < 1e-6:
            print(f"    WARNING: Deformation field is essentially zero! This may explain perfect Dice scores if input segmentations are already aligned.")

        # Calculate metrics if segmentation is available
        if has_segmentation:
            # Apply transformation to segmentation
            warped_seg = utils.numpy_im(seg_stn(moving_seg_torch, pred), 1, device)
            fixed_seg_numpy = utils.numpy_im(fixed_seg, 1)
            
            # Round to nearest integer for label-wise evaluation
            warped_seg = np.round(warped_seg).astype(np.int32)
            fixed_seg_numpy = np.round(fixed_seg_numpy).astype(np.int32)
            
            # Per-organ Dice computation using the score_case function
            result = score_case(fixed_seg_numpy, warped_seg, label_list)
            dices.append(result['DICE'])
            
            # Overall Dice for backwards compatibility (binary)
            overall_dice = np.mean(result['DICE'])
            
            # Calculate HD95 if medpy is available (using binary masks)
            if MEDPY_AVAILABLE:
                try:
                    hd95_score = hd95(warped_seg > 0, fixed_seg_numpy > 0)
                    hd95s.append(hd95_score)
                except Exception as e:
                    print(f"Warning: HD95 calculation failed for sample {i}: {e}")
                    hd95s.append(np.nan)
            else:
                hd95s.append(np.nan)
        else:
            # No segmentation available
            dices.append(np.zeros(len(label_list)))  # Append zeros for all organs
            hd95s.append(np.nan)

        # Save outputs if requested
        if args.save_warped:
            # Save warped image
            save_nifti(warped_im, os.path.join(cfg.TEST_PATH, f'{i}_warped_image.nii.gz'))
            
            # Save deformation field
            save_nifti(flow_numpy, os.path.join(cfg.TEST_PATH, f'{i}_deformation_field.nii.gz'))
            
            if has_segmentation:
                # Save warped segmentation
                save_nifti(warped_seg, os.path.join(cfg.TEST_PATH, f'{i}_warped_seg.nii.gz'))

        # Save visualization images
        if i % 1 == 0:
            cv2.imwrite('{}/{}-warped.bmp'.format(cfg.TEST_PATH, i), warped_im[:, :, idx])
            
            # Save flow visualization
            flow_vis = utils.render_flow(flow_numpy[:, :, :, idx])
            cv2.imwrite('{}/{}-flow.png'.format(cfg.TEST_PATH, i), flow_vis)
            
            if has_segmentation:
                # Save segmentation overlays
                vis_seg = utils.render_image_with_mask(
                    utils.numpy_im(fixed, device=device)[:, :, idx], 
                    warped_seg[:, :, idx], color=1)
                cv2.imwrite('{}/{}-vis_warped.png'.format(cfg.TEST_PATH, i), vis_seg)
                
                vis_seg_gt = utils.render_image_with_mask(
                    utils.numpy_im(fixed, device=device)[:, :, idx], 
                    fixed_seg_numpy[:, :, idx], color=0)
                cv2.imwrite('{}/{}-vis_gt.png'.format(cfg.TEST_PATH, i), vis_seg_gt)

                # Save middle slice overlays and error maps
                # Save overlay: fixed CT + label
                save_overlay_png(fixed_np, fixed_seg_numpy, os.path.join(cfg.TEST_PATH, f"{i}_fixed_CT_overlay.png"))
                # Save overlay: moving MR + label
                save_overlay_png(moving_np, moving_seg, os.path.join(cfg.TEST_PATH, f"{i}_moving_MR_overlay.png"))
                # Save overlay: warped MR + warped label
                save_overlay_png(warped_im, warped_seg, os.path.join(cfg.TEST_PATH, f"{i}_warped_MR_overlay.png"))
                # Save deformation field visualization
                save_deformation_field_png(flow_for_jac, os.path.join(cfg.TEST_PATH, f"{i}_deformation_field_middle.png"))
                # Save error map
                save_error_map_png(warped_seg, fixed_seg_numpy, os.path.join(cfg.TEST_PATH, f"{i}_error_map_middle.png"))

        # Print progress
        if has_segmentation and len(dices[i]) > 0 and np.any(dices[i] > 0):
            overall_dice = np.mean(dices[i])
            hd95_str = f", HD95: {hd95s[i]:.2f}" if not np.isnan(hd95s[i]) else ", HD95: N/A"
            print(f'Sample {i}: Overall Dice: {overall_dice:.4f}{hd95_str}, Neg Jac: {neg_jac_percent:.2f}%, Time: {inference_time:.4f}s')
            
            # Print per-organ metrics
            organ_metrics_str = ", ".join([f"Organ {label}: {dices[i][idx]:.4f}" 
                                         for idx, label in enumerate(label_list)])
            print(f"    Per-organ Dice -> {organ_metrics_str}")
        else:
            print(f'Sample {i}: Neg Jac: {neg_jac_percent:.2f}%, Time: {inference_time:.4f}s')
        
        if i == 16:  # Limit to 17 samples for testing
            break

    # Calculate and print final statistics (following baseline dataset pattern)
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    # Filter out cases without segmentation (zeros arrays)
    valid_dices = [d for d in dices if isinstance(d, np.ndarray) and np.any(d > 0)]
    valid_hd95s = [h for h in hd95s if not np.isnan(h)]
    
    if valid_dices:
        # Convert to array for per-organ analysis
        dice_array = np.array(valid_dices)
        
        # Overall metrics
        overall_dice_mean = np.mean(valid_dices)
        overall_dice_std = np.std(valid_dices)
        
        # Per-organ metrics (following baseline dataset pattern)
        dice_mean_by_organ = np.nanmean(dice_array, axis=0)
        dice_std_by_organ = np.nanstd(dice_array, axis=0)
        
        print(f"Overall DICE: {overall_dice_mean:.4f} ± {overall_dice_std:.4f}")
        print(f"DICE mean by organ: {dice_mean_by_organ}")
        print(f"DICE std by organ: {dice_std_by_organ}")
        
        print(f"\nPer-Organ Dice Results:")
        for idx, label in enumerate(label_list):
            print(f"  Organ {label}: {dice_mean_by_organ[idx]:.4f} ± {dice_std_by_organ[idx]:.4f}")
            
        print(f"\nSample count: {len(valid_dices)}")
    else:
        print("Dice Score: N/A (no segmentation data)")
    
    if valid_hd95s and MEDPY_AVAILABLE:
        print(f"HD95: {np.mean(valid_hd95s):.2f} ± {np.std(valid_hd95s):.2f} mm (n={len(valid_hd95s)})")
    else:
        print("HD95: N/A (no segmentation data or medpy not available)")
    
    print(f"Negative Jacobian %: {np.mean(neg_jac_percentages):.4f} ± {np.std(neg_jac_percentages):.4f}")
    print(f"Inference Time: {np.mean(inference_times):.4f} ± {np.std(inference_times):.4f} seconds")
    print(f"Total Parameters: {total_params:,}")
    
    # Save per-organ Dice scores to CSV
    save_dice_csv(dices, label_list, os.path.join(cfg.TEST_PATH, 'per_organ_dice_scores.csv'))
    
    print("="*60)














