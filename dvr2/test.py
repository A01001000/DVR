import torch
import torch.nn.functional as F
import numpy as np
import time
import nibabel as nib
import os
from medpy.metric.binary import dc, hd95 # Assuming you use medpy for metrics
import torchvision
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast 
from tqdm import tqdm


from dvr2.models_rob import VLearnPolicy_Medium, SpatialTransformer, AdaptiveFlowPolicy_Medium
from dvr2.dino_encoder import DINOEncoder
from dvr2.utils import MINDLoss, apply_flow_scaling
from dvr2.student_extractor import StudentEncoder3D
from dvr2.extract_dino import extract_single_dino_features

def precompute_features_in_memory(data_loader, dino_encoder, pca_transformer, device):
    """
    Revised: Iterates through a DataLoader, runs the student extractor,
    and caches the images and features in a list in RAM.
    """
    cached_data = []
    # student_extractor.eval()
    
    with torch.no_grad():
        # The DataLoader now yields downsampled images directly
        for mr_img, ct_img, spacing in tqdm(data_loader, desc="Pre-computing features"):
            
            mr_img = mr_img.to(device)
            ct_img = ct_img.to(device)

            # Extract features with the student model
            # Note: The model expects a batch dimension, so we add it if it's missing
            # f_mr = student_extractor(mr_img if mr_img.dim() == 5 else mr_img.unsqueeze(0))
            # f_ct = student_extractor(ct_img if ct_img.dim() == 5 else ct_img.unsqueeze(0))
            f_mr, f_ct = extract_single_dino_features(dino_encoder, mr_img.squeeze(), ct_img.squeeze(), pca_transformer)

            # Move results to CPU to store in the list, saving VRAM
            # Squeeze to remove the batch dimension
            cached_data.append((
                mr_img.squeeze(0).cpu(),
                ct_img.squeeze(0).cpu(),
                spacing,
                f_mr.squeeze(0).cpu(), 
                f_ct.squeeze(0).cpu(),
            ))
            
    return cached_data

def jacobian_determinant(flow):
    """
    Calculates the determinant of the Jacobian of a 3D deformation field.
    The deformation field is the flow field. The Jacobian of the transformation phi
    is J(phi) = I + J(u), where u is the displacement field (flow).
    
    Args:
        flow (torch.Tensor): A 5D tensor of shape (B, 3, D, H, W), where 3
                             represents the displacement vectors (u_x, u_y, u_z).
    
    Returns:
        torch.Tensor: A 4D tensor of shape (B, D, H, W) representing the
                      Jacobian determinant at each voxel.
    """
    # Define the spatial dimensions we want gradients for
    spatial_dims = (1, 2, 3)

    # Gradients of the x-component of the flow
    # FIX: Specify the dimensions to calculate gradients along
    grad_u_z, grad_u_y, grad_u_x = torch.gradient(flow[:, 0, ...], dim=spatial_dims)
    
    # Gradients of the y-component of the flow
    # FIX: Specify the dimensions
    grad_v_z, grad_v_y, grad_v_x = torch.gradient(flow[:, 1, ...], dim=spatial_dims)
    
    # Gradients of the z-component of the flow
    # FIX: Specify the dimensions
    grad_w_z, grad_w_y, grad_w_x = torch.gradient(flow[:, 2, ...], dim=spatial_dims)

    # Assemble the Jacobian matrix J(u) and add the identity matrix
    Jxx = grad_u_x + 1
    Jxy = grad_u_y
    Jxz = grad_u_z

    Jyx = grad_v_x
    Jyy = grad_v_y + 1
    Jyz = grad_v_z

    Jzx = grad_w_x
    Jzy = grad_w_y
    Jzz = grad_w_z + 1

    # Compute the determinant
    determinant = Jxx * (Jyy * Jzz - Jyz * Jzy) \
                - Jxy * (Jyx * Jzz - Jyz * Jzx) \
                + Jxz * (Jyx * Jzy - Jyy * Jzx)
    
    return determinant.unsqueeze(1)

def visualize_results(fixed_vol, moving_vol, warped_vol, flow, save_dir, patient_id):
    """
    Saves the middle slice of the registration and the deformation field.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Save a visual comparison slice
    if fixed_vol.ndim == 4: 
        fixed_vol = fixed_vol.unsqueeze(1)
        moving_vol = moving_vol.unsqueeze(1)
    slice_idx = fixed_vol.shape[2] // 2 # Get middle slice
    
    # Normalize images to [0, 1] for saving as PNG
    fixed_slice = (fixed_vol[0, 0, slice_idx, :, :] - fixed_vol.min()) / (fixed_vol.max() - fixed_vol.min())
    moving_slice = (moving_vol[0, 0, slice_idx, :, :] - moving_vol.min()) / (moving_vol.max() - moving_vol.min())
    warped_slice = (warped_vol[0, 0, slice_idx, :, :] - warped_vol.min()) / (warped_vol.max() - warped_vol.min())
    
    # Create a single comparison image (Fixed | Warped | Moving)
    comparison_grid = torch.cat([fixed_slice, warped_slice, moving_slice], dim=1)
    
    save_path_img = os.path.join(save_dir, f"patient_{patient_id}_comparison.png")
    torchvision.utils.save_image(comparison_grid, save_path_img)
    print(f"Saved visualization to {save_path_img}")
    
    # 2. Save the deformation field as a .nii.gz file
    # This can be opened in viewers like ITK-SNAP or 3D Slicer
    save_path_flow = os.path.join(save_dir, f"patient_{patient_id}_flow.nii.gz")
    
    # Flow should be [H, W, D, 3] for Nifti, so permute from [3, D, H, W]
    flow_to_save = flow.squeeze(0).cpu().numpy().transpose(2, 3, 1, 0)
    nifti_flow = nib.Nifti1Image(flow_to_save, np.eye(4)) # Using identity affine
    nib.save(nifti_flow, save_path_flow)
    print(f"Saved deformation field to {save_path_flow}")

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
    """Compute per-organ metrics for a single case."""
    dice_coefficient = compute_label_wise_dice(seg_fixed, seg_moving, label_list)
    dice_coefficient = np.array(dice_coefficient)
    
    # For now, we'll focus on Dice. HD95 can be added later if needed
    return {'DICE': dice_coefficient}


def detect_dataset_labels(test_loader):
    """Detect available labels in the dataset by examining actual segmentation data."""
    all_labels = set()
    samples_checked = 0
    max_samples_to_check = min(10, len(test_loader.dataset))
    
    for batch_idx, (_, _, _, mr_seg_batch, ct_seg_batch) in enumerate(test_loader):
        if samples_checked >= max_samples_to_check:
            break
            
        # Check both MR and CT segmentations in the batch
        for seg_batch in [mr_seg_batch, ct_seg_batch]:
            for i in range(seg_batch.shape[0]):
                seg_data = seg_batch[i].cpu().numpy()
                labels = np.unique(seg_data)
                non_zero_labels = labels[labels > 0]
                all_labels.update(non_zero_labels.astype(int))
                samples_checked += 1
                
                if samples_checked >= max_samples_to_check:
                    break
            if samples_checked >= max_samples_to_check:
                break
    
    if not all_labels:
        # Default fallback - assume common organ labels
        return [1, 2, 3, 4]  # liver, kidney, pancreas, spleen
    
    detected_labels = sorted(list(all_labels))
    print(f"Detected labels from {samples_checked} samples: {detected_labels}")
    return detected_labels


def evaluate_vlearn(test_loader, save_dir, vol_shape, feature_shape, model_dir, device, max_steps=5):
    """
    Evaluates the iterative VLearn policy in a vectorized (batched) manner.
    """
    # Check if test loader has any data
    if len(test_loader) == 0:
        print("⚠️  Warning: Empty test loader. Returning default metrics.")
        return {
            "dice": 0.0,
            "dice_std": 0.0,
            "hd95": 0.0,
            "hd95_std": 0.0,
            "jacobian_neg_percent": 0.0,
            "inference_time": 0.0,
            "num_parameters": 0
        }
    
    # --- Model and Helper Initialization ---
    IMG_SIZE = 128
    
    # Paths
    model_path = os.path.join(model_dir, "best_vlearn_model.pth")
    # Handle both 2D and 3D vol_shape
    if len(vol_shape) == 3:
        D, H_vol, W_vol = vol_shape
    elif len(vol_shape) == 2:
        H_vol, W_vol = vol_shape
        D = 1  # Default depth for 2D data
        print(f"Warning: 2D vol_shape detected {vol_shape}, assuming depth=1")
    else:
        raise ValueError(f"Unexpected vol_shape dimensions: {vol_shape}")
        
    C_feat, D_feat, H_feat, W_feat = feature_shape
    
    # Load checkpoint to determine model architecture
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    hidden_dims = [16, 32, 64, 128, 256] # Default fallback
    print("Using medium [16, 32, 64, 128, 256] network size for evaluation.")

    # Check if the checkpoint contains twin critics
    has_twin_critics = any('critic1' in key or 'critic2' in key for key in checkpoint.keys())
    print(f"Twin critics detected in checkpoint: {has_twin_critics}")
    
    policy = AdaptiveFlowPolicy_Medium(
        in_channels=(C_feat * 2) + 3,
        use_checkpointing=False  # Disable for evaluation
    ).to(device)
    
       # Load trained Student Feature Extractor ---
    print("🧠 Loading trained student feature extractor...")
    student_extractor = StudentEncoder3D(in_channels=1, out_channels=feature_shape[0]) # out_channels=64
    student_extractor.load_state_dict(torch.load(os.path.join(model_dir, "student_feature_extractor.pth")))
    student_extractor.to(device)
    student_extractor.eval() # IMPORTANT: Set to evaluation mode
    
    print(f"Created VLearnPolicy with architecture: {hidden_dims}")
    print(f"Twin critics in checkpoint: {has_twin_critics}")
    print(f"Model parameters: {sum(p.numel() for p in policy.parameters())}")
    
    # Handle loading with potential architecture mismatches
    # 1. Load the state dict from the checkpoint file
    checkpoint_path = os.path.join(model_dir, "best_vlearn_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 1. Extract the nested state_dict
    # This is the crucial step you were missing.
    state_dict = checkpoint['policy_state_dict']

    # 2. Create a new, clean state dictionary to handle 'torch.compile' prefixes
    clean_state_dict = {}
    for k, v in state_dict.items():
        # Remove the '_orig_mod.' prefix if the model was compiled
        if k.startswith('_orig_mod.'):
            clean_state_dict[k[10:]] = v
        else:
            clean_state_dict[k] = v
                
    # 3. Load the clean and correct state dictionary into the policy
    policy.load_state_dict(clean_state_dict)
    print(f"Model loaded successfully. Trainable parameters: {sum(p.numel() for p in policy.parameters() if p.requires_grad)}")
    
    # Check if model has twin critics
    has_twin_critics = hasattr(policy, 'critic1') and hasattr(policy, 'critic2')
    print(f"Twin critics detected: {has_twin_critics}")
    if has_twin_critics:
        print("✅ Model has twin critics - using minimum value for evaluation")
    else:
        print("⚠️  Model has single critic - using standard evaluation")
    policy.eval()
    transformer = SpatialTransformer().to(device)

    # Detect available labels for per-organ evaluation
    print("Detecting dataset labels...")
    label_list = detect_dataset_labels(test_loader)
    print(f"Will evaluate per-organ Dice for labels: {label_list}")

    # --- Metric and Result Trackers ---
    dice_list = []  # Will store per-organ Dice arrays for each patient
    hd95_list = []  # For binary HD95 scores
    jacobian_list = []
    inference_time_list = []
    
    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    
    # --- Evaluation Loop ---
    # The loop now processes an entire batch in each iteration
    for batch_idx, (mr_vol_batch, ct_vol_batch, batch_spacing, mr_seg_batch, ct_seg_batch) in enumerate(test_loader):
        print(f"\nEvaluating batch {batch_idx+1}/{len(test_loader)}...")
        
        # Move the entire batch of tensors to the device
        mr_vol_batch, ct_vol_batch = mr_vol_batch.to(device), ct_vol_batch.to(device)
        mr_seg_batch, ct_seg_batch = mr_seg_batch.to(device), ct_seg_batch.to(device)
        
        with torch.no_grad():
            start_time = time.time()
            
            # Extract features
            f_mr_batch = student_extractor(mr_vol_batch)
            f_ct_batch = student_extractor(ct_vol_batch)
            
            # Get batch size dynamically from the tensor
            batch_size = f_mr_batch.shape[0]
            
            current_spacing = batch_spacing

            # Initialize flow_acc with the correct batch size
            flow_acc = torch.zeros((batch_size, 3, D, H_vol, W_vol), device=device)

            # The features from the student_extractor are already in the correct [B, C, D, H, W] format
            f_mr_to_warp = f_mr_batch 
            f_ct_fixed = f_ct_batch

            feature_map_spatial_dims = f_mr_batch.shape[1:]

            for step in range(max_steps):
                # FIX: Downsample the accumulated flow to the feature map's resolution
                # Downsample the accumulated flow to the feature map's resolution
                feature_map_size = f_ct_fixed.shape[2:] 
                flow_for_features = F.interpolate(flow_acc, size=feature_map_size, mode='trilinear', align_corners=False)
                flow_for_features = apply_flow_scaling(flow_for_features, flow_acc.shape[2:], feature_map_size, current_spacing)
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # No .unsqueeze(0) or .squeeze(0) needed. All ops are batched.
                    # Warp the batch of moving features
                    warped_f_mr_batch = transformer(f_mr_to_warp, flow_for_features)
                    
                    # Create the state for the whole batch
                    current_state = torch.cat([
                        warped_f_mr_batch, 
                        f_ct_fixed, 
                        flow_for_features
                    ], dim=1)
                    
                    # Get the deterministic action (the mean) for the whole batch
                    action_dist = policy.get_action_dist(current_state)
                    incremental_flow_low_res = action_dist.mean

                    # Debug: Check incremental flow on first step
                    if step == 0:
                        print(f"  Step {step} - Incremental flow stats - Min: {incremental_flow_low_res.min().item():.4f}, Max: {incremental_flow_low_res.max().item():.4f}, Mean: {incremental_flow_low_res.mean().item():.4f}")

                # Upsample the incremental flow for the whole batch
                incremental_flow = F.interpolate(
                        incremental_flow_low_res,
                        size=flow_acc.shape[2:], # Target the full resolution
                        mode='trilinear',
                        align_corners=False
                )
                incremental_flow_full_res = apply_flow_scaling(incremental_flow_full_res, incremental_flow_low_res.shape[1:], flow_acc.shape[2:])

                flow_acc += incremental_flow

            inference_time = (time.time() - start_time) / batch_size # Time per sample
            
            print("Upsampling final low-resolution flow field to full resolution for warping...")
            original_image_size = mr_vol_batch.shape[1:] # Get shape [192, 192, 192] from the data
            
            # flow_to_resize = flow_acc.unsqueeze(0) # Shape -> [1, 3, D_low, H_low, W_low]

            # 2. Get only the SPATIAL dimensions [D, H, W] of the target volume
            target_spatial_size = mr_vol_batch.shape[2:]

            # 3. Call interpolate with correctly shaped tensors
            full_res_flow = F.interpolate(
                flow_acc,
                size=target_spatial_size,
                mode='trilinear',
                align_corners=False
            )
            full_res_flow = apply_flow_scaling(full_res_flow , flow_acc.shape[2:], target_spatial_size, current_spacing)
            
            # Debug: Check deformation field statistics
            print(f"  Flow field stats - Min: {flow_acc.min().item():.4f}, Max: {flow_acc.max().item():.4f}, Mean: {flow_acc.mean().item():.4f}")
            print(f"  Flow field has NaN: {torch.isnan(flow_acc).any().item()}")
            print(f"  Flow field has Inf: {torch.isinf(flow_acc).any().item()}")
            
            # Additional debugging: Check flow magnitudes
            flow_magnitude = torch.norm(flow_acc, dim=1)
            print(f"  Flow magnitude stats - Min: {flow_magnitude.min().item():.4f}, Max: {flow_magnitude.max().item():.4f}, Mean: {flow_magnitude.mean().item():.4f}")
            
            # Check if flow values are reasonable (should be much smaller than image dimensions)
            print(f"  Image dimensions: D={D}, H={H_vol}, W={W_vol}")
            print(f"  Flow as percentage of image size - X: {(flow_acc[:, 0].abs().max() / W_vol * 100).item():.1f}%, Y: {(flow_acc[:, 1].abs().max() / H_vol * 100).item():.1f}%, Z: {(flow_acc[:, 2].abs().max() / D * 100).item():.1f}%")
            
            # TESTING CORRECTED SPATIAL TRANSFORMER
            print(f"  *** TESTING CORRECTED SPATIAL TRANSFORMER ***")
            
            # --- Post-processing: Calculate metrics and visualize for each sample in the batch ---
            # Warp the entire batch of segmentations and volumes at once
            print(f"  Pre-warp segmentation stats - Min: {mr_seg_batch.min().item()}, Max: {mr_seg_batch.max().item()}, Sum: {mr_seg_batch.sum().item()}")
            print("mr_seg_batch shape:", mr_seg_batch.shape)
            print("mr_vol_batch shape:", mr_vol_batch.shape)
            warped_seg_batch = transformer(mr_seg_batch.float(), full_res_flow, mode='nearest')
            warped_vol_batch = transformer(mr_vol_batch, full_res_flow)
            
            print(f"  Post-warp segmentation stats - Min: {warped_seg_batch.min().item()}, Max: {warped_seg_batch.max().item()}, Sum: {warped_seg_batch.sum().item()}")
            
            jac_det_batch = jacobian_determinant(full_res_flow)
            
            # Loop at the END to calculate metrics and save images for each sample
            for i in range(batch_size):
                inference_time_list.append(inference_time)
                patient_id = batch_idx * test_loader.batch_size + i + 1
                try:
                    # Add a sanity check print statement
                    print(f"  Sum of original segmentation: {mr_seg_batch[i].sum().item()}")
                    print(f"  Sum of warped segmentation: {warped_seg_batch[i].sum().item()}")

                    # Convert to NumPy and calculate metrics
                    warped_seg_np = warped_seg_batch[i].squeeze().cpu().numpy()
                    ct_seg_np = ct_seg_batch[i].squeeze().cpu().numpy()
                    
                    # Round to nearest integer for label-wise evaluation
                    warped_seg_np = np.round(warped_seg_np).astype(np.int32)
                    ct_seg_np = np.round(ct_seg_np).astype(np.int32)

                    # Compute per-organ metrics using score_case function
                    result = score_case(ct_seg_np, warped_seg_np, label_list)
                    dice_list.append(result['DICE'])
                    
                    # Overall Dice (binary) for backwards compatibility
                    warped_seg_binary = warped_seg_np > 0.5
                    ct_seg_binary = ct_seg_np > 0.5
                    hd95_score = hd95(warped_seg_binary, ct_seg_binary)
                    hd95_list.append(hd95_score)
                    
                    jac_det = jacobian_determinant(flow_acc[i:i+1]) # Pass a 5D tensor
                    neg_jac_percent = (jac_det.squeeze().cpu().numpy() < 0).mean() * 100
                    jacobian_list.append(neg_jac_percent)
                
                    # Print overall metrics
                    overall_dice = np.mean(result['DICE'])
                    print(f"  Patient {patient_id} -> Overall Dice: {overall_dice:.4f}, HD95: {hd95_score:.2f}, Neg Jac: {neg_jac_percent:.2f}%")
                    
                    # Print per-organ metrics
                    organ_metrics_str = ", ".join([f"Organ {label}: {result['DICE'][idx]:.4f}" 
                                                 for idx, label in enumerate(label_list)])
                    print(f"    Per-organ Dice -> {organ_metrics_str}")

                except RuntimeError as e:
                    print(f"  !! METRIC CALCULATION FAILED for Patient {patient_id}: {e}")
                    # Append placeholder values so the logs stay aligned
                    dice_list.append(np.zeros(len(label_list)))
                    hd95_list.append(999) # A high value for failed HD
                    jacobian_list.append(100)
                
                visualize_results(
                    fixed_vol=ct_vol_batch[i:i+1],      # Slicing keeps the batch dim, making it 4D
                    moving_vol=mr_vol_batch[i:i+1],    # Slicing keeps the batch dim, making it 4D
                    warped_vol=warped_vol_batch[i:i+1],  # This is already 5D, slicing keeps it 5D
                    flow=flow_acc[i:i+1],              # This is already 5D, slicing keeps it 5D
                    save_dir=save_dir, 
                    patient_id=patient_id
                )

    # --- Aggregate and Return Final Metrics (following the pattern from baseline datasets) ---
    # Create arrays for analysis with safety checks
    dice_array = np.array(dice_list) if dice_list else np.array([])
    hd95_array = np.array(hd95_list) if hd95_list else np.array([])
    
    # Calculate overall metrics with safety checks
    if len(dice_list) > 0:
        overall_dice_mean = np.mean(dice_list)
        overall_dice_std = np.std(dice_list)
    else:
        overall_dice_mean = 0.0
        overall_dice_std = 0.0
        print("⚠️  Warning: No dice scores available")
    
    # Calculate per-organ metrics with safety checks
    if dice_array.size > 0 and dice_array.ndim > 1:
        dice_mean_by_organ = np.nanmean(dice_array, axis=0)
        dice_std_by_organ = np.nanstd(dice_array, axis=0)
        
        # Handle case where mean returns scalar instead of array
        if np.isscalar(dice_mean_by_organ):
            dice_mean_by_organ = np.array([dice_mean_by_organ])
        if np.isscalar(dice_std_by_organ):
            dice_std_by_organ = np.array([dice_std_by_organ])
    else:
        # No per-organ data available
        dice_mean_by_organ = np.array([])
        dice_std_by_organ = np.array([])
        print("⚠️  Warning: No per-organ dice data available")
    
    # HD95 and other metrics with safety checks
    if len(hd95_list) > 0:
        hd95_mean = np.nanmean(hd95_array)
        hd95_std = np.nanstd(hd95_array)
    else:
        hd95_mean = 0.0
        hd95_std = 0.0
        print("⚠️  Warning: No HD95 scores available")
    
    if len(jacobian_list) > 0:
        jacobian_mean = np.mean(jacobian_list)
    else:
        jacobian_mean = 0.0
        print("⚠️  Warning: No Jacobian scores available")
    
    if len(inference_time_list) > 0:
        inference_time_mean = np.mean(inference_time_list)
    else:
        inference_time_mean = 0.0
        print("⚠️  Warning: No inference time data available")
    
    # Create final metrics dictionary for compatibility
    final_metrics = {
        "dice": overall_dice_mean,
        "dice_std": overall_dice_std,
        "hd95": hd95_mean,
        "hd95_std": hd95_std,
        "jacobian_neg_percent": jacobian_mean,
        "inference_time": inference_time_mean,
        "num_parameters": num_params
    }
    
    # Add per-organ metrics with safety checks
    if isinstance(dice_mean_by_organ, np.ndarray) and dice_mean_by_organ.size > 0 and len(label_list) > 0:
        for idx, label in enumerate(label_list):
            if idx < len(dice_mean_by_organ):
                final_metrics[f"organ_{label}_dice"] = dice_mean_by_organ[idx]
                if isinstance(dice_std_by_organ, np.ndarray) and idx < len(dice_std_by_organ):
                    final_metrics[f"organ_{label}_dice_std"] = dice_std_by_organ[idx]
                else:
                    final_metrics[f"organ_{label}_dice_std"] = 0.0
            else:
                print(f"⚠️  Warning: No dice data for organ {label}")
                final_metrics[f"organ_{label}_dice"] = 0.0
                final_metrics[f"organ_{label}_dice_std"] = 0.0
    else:
        print("⚠️  Warning: No per-organ metrics to assign")
    
    print(f"\nFinal Aggregated Metrics:")
    print(f"  Overall - Dice: {overall_dice_mean:.4f} ± {overall_dice_std:.4f}")
    print(f"  HD95: {hd95_mean:.2f} ± {hd95_std:.2f}")
    print(f"  Jacobian: {jacobian_mean:.2f}%, Time: {inference_time_mean:.4f}s")
    print(f"  Parameters: {num_params}")
    
    print(f"\nPer-Organ Dice Results:")
    if isinstance(dice_mean_by_organ, np.ndarray) and dice_mean_by_organ.size > 0 and len(label_list) > 0:
        for idx, label in enumerate(label_list):
            if idx < len(dice_mean_by_organ) and idx < len(dice_std_by_organ):
                print(f"  Organ {label}: {dice_mean_by_organ[idx]:.4f} ± {dice_std_by_organ[idx]:.4f}")
            else:
                print(f"  Organ {label}: No data available")
    else:
        print("  No per-organ results available")
    
    return final_metrics

def validate_memory_efficient(policy, val_loader, device, vol_shape, feature_shape, max_steps=5, max_validation_samples=None):
    """
    Memory-efficient validation that processes smaller batches and uses CPU offloading.
    
    Args:
        max_validation_samples: Limit number of validation samples (None = use all)
    
    Returns:
        Mean NCC score across validation samples
    """
    policy.eval()
    transformer = SpatialTransformer().to(device)
    mind_loss_func = MINDLoss().cpu() 
    all_mind_scores = []
    
    D_vol, H_vol, W_vol = vol_shape
    C_feat, D_feat, H_feat, W_feat = feature_shape
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (mr_vol_batch, ct_vol_batch, batch_spacing, f_mr_batch, f_ct_batch) in enumerate(val_loader):
            # MEMORY OPTIMIZATION 1: Process one sample at a time from each batch
            batch_size = f_mr_batch.shape[0]
            
            for sample_idx in range(batch_size):
                if max_validation_samples and sample_count >= max_validation_samples:
                    break
                
                # Extract single sample
                f_mr = f_mr_batch[sample_idx:sample_idx+1].to(device)
                f_ct = f_ct_batch[sample_idx:sample_idx+1].to(device)
                mr_vol = mr_vol_batch[sample_idx:sample_idx+1].to(device)
                ct_vol = ct_vol_batch[sample_idx:sample_idx+1].to(device)
                
                if mr_vol.shape[2:] != vol_shape:
                    mr_vol = F.interpolate(mr_vol, size=vol_shape, mode='trilinear', align_corners=False)
                    ct_vol = F.interpolate(ct_vol, size=vol_shape, mode='trilinear', align_corners=False)
                    f_mr = F.interpolate(f_mr, size=feature_shape[1:], mode='trilinear', align_corners=False)
                    f_ct = F.interpolate(f_ct, size=feature_shape[1:], mode='trilinear', align_corners=False)

                current_spacing = batch_spacing[sample_idx]

                # Reshape features for single sample
                f_mr_reshaped = f_mr.view(1, C_feat, D_feat, H_feat, W_feat) # Simpler reshape
                f_ct_reshaped = f_ct.view(1, C_feat, D_feat, H_feat, W_feat) 
                flow_acc = torch.zeros((1, 3, D_vol, H_vol, W_vol), device=device)
                feature_map_spatial_dims = f_ct_reshaped.shape[2:] # (D_feat, H_feat, W_feat)
                
                # Registration loop with memory cleanup
                for step in range(max_steps):
                    feature_map_size = f_ct_reshaped.shape[2:]
                    flow_for_features = F.interpolate(flow_acc, size=feature_map_size, mode='trilinear', align_corners=False)
                    flow_for_features = apply_flow_scaling(flow_for_features, flow_acc.shape[2:], feature_map_spatial_dims, current_spacing)
                    
                    warped_f_mr = transformer(f_mr_reshaped, flow_for_features)
                    
                    # Create the complete, "deformation-aware" state with 131 channels
                    current_state = torch.cat([
                        warped_f_mr, 
                        f_ct_reshaped, 
                        flow_for_features # Add the flow here
                    ], dim=1)
                    
                    # Get the deterministic action from the policy
                    action_dist = policy.get_action_dist(current_state)
                    incremental_flow_low_res = action_dist.mean
                    
                    incremental_flow = F.interpolate(
                        incremental_flow_low_res,
                        size=flow_acc.shape[2:],  # Target the full resolution of flow_acc
                        mode='trilinear',
                        align_corners=False
                    )
                    incremental_flow = apply_flow_scaling(incremental_flow, incremental_flow_low_res.shape[1:], flow_acc.shape[2:])
    
                    # Upsample and accumulate the flow
                    flow_acc += incremental_flow
    
                    # Clean up intermediate tensors
                    del warped_f_mr, current_state, incremental_flow, incremental_flow_low_res, flow_for_features
                
                # Warp image and compute similarity
                warped_mr = transformer(mr_vol, flow_acc)
                
                # MEMORY OPTIMIZATION 2: Move to CPU for metric calculation
                warped_mr_cpu = warped_mr.cpu()
                ct_vol_cpu = ct_vol.cpu()
                
                warped_mr_lowres = F.interpolate(warped_mr_cpu, scale_factor=0.5, mode='trilinear')
                ct_vol_lowres = F.interpolate(ct_vol_cpu, scale_factor=0.5, mode='trilinear')
    
                # Compute MIND score on CPU
                mind_score = mind_loss_func(warped_mr_lowres, ct_vol_lowres).item()
                all_mind_scores.append(mind_score)
                
                # Clean up GPU tensors
                del f_mr, f_ct, mr_vol, ct_vol, f_mr_reshaped, f_ct_reshaped, flow_acc, warped_mr
                del warped_mr_cpu, ct_vol_cpu, warped_mr_lowres, ct_vol_lowres

                sample_count += 1
                
                # MEMORY OPTIMIZATION 3: Periodic GPU cache cleanup
                if sample_count % 5 == 0:
                    torch.cuda.empty_cache()
            
            # Break if we've reached max samples
            if max_validation_samples and sample_count >= max_validation_samples:
                break
    
    # Final cleanup
    torch.cuda.empty_cache()
    policy.train()  # Set back to training mode

    mean_mind = np.mean(all_mind_scores) if all_mind_scores else 0
    print(f"   Memory-efficient validation (n={sample_count}) - MIND: {mean_mind:.4f}")

    return mean_mind

def compute_ncc_cpu(img1, img2):
    """CPU-based NCC computation to save GPU memory during validation"""
    # Convert to numpy if needed
    if hasattr(img1, 'numpy'):
        img1 = img1.numpy()
    if hasattr(img2, 'numpy'):
        img2 = img2.numpy()
    
    # Flatten and compute means
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    mean1 = np.mean(img1_flat)
    mean2 = np.mean(img2_flat)
    
    # Center the data
    img1_centered = img1_flat - mean1
    img2_centered = img2_flat - mean2
    
    # Compute NCC
    numerator = np.sum(img1_centered * img2_centered)
    denominator = np.sqrt(np.sum(img1_centered**2) * np.sum(img2_centered**2))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator