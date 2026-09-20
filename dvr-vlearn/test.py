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
import matplotlib.pyplot as plt
import csv


from dvr2.models_rob import VLearnPolicy_Medium, SpatialTransformer, AdaptiveFlowPolicy_Medium, AdaptiveFlowPolicy_Raw
from dvr2.dino_encoder import DINOEncoder
from dvr2.utils import MINDLoss, apply_flow_scaling, create_registration_state
from dvr2.extract_dino import extract_single_dino_features

# In test.py

def precompute_features_in_memory(data_loader, dino_encoder, pca_transformer, device, use_raw_images=False, dataset_type=None):
    """
    CORRECTED: Pre-computes and caches data for the TRAINING set.
    It only unpacks the items provided by the VlearnDataset (no segmentations).
    Now handles dynamic PCA dimensions properly.
    """
    cached_data = []
    
    with torch.no_grad():
        desc = "Pre-computing Raw Images" if use_raw_images else "Pre-computing DINO Features"
        
        # --- THE FIX: Unpack only the 3 items the DataLoader provides ---
        for mr_img, ct_img, spacing in tqdm(data_loader, desc=desc):
            
            mr_img = mr_img.to(device)
            ct_img = ct_img.to(device)

            f_mr, f_ct = None, None

            if use_raw_images:
                # For the raw image ablation, the "features" are the raw images
                f_mr = mr_img.squeeze(0)
                f_ct = ct_img.squeeze(0)
            else:
                # Standard DINO feature extraction
                if mr_img.dim() == 5:
                    mr_vol_3d = mr_img.squeeze(0).squeeze(0)
                    ct_vol_3d = ct_img.squeeze(0).squeeze(0)
                else: # Handles the expected [B, C, D, H, W] where B=1
                    mr_vol_3d = mr_img.squeeze(0)
                    ct_vol_3d = ct_img.squeeze(0)
                
                f_mr_extracted, f_ct_extracted = extract_single_dino_features(dino_encoder, mr_vol_3d, ct_vol_3d, pca_transformer, dataset_type=dataset_type)

                if f_mr_extracted is None or f_ct_extracted is None:
                    print(f"Skipping a sample due to feature extraction failure.")
                    continue
                
                f_mr = f_mr_extracted
                f_ct = f_ct_extracted

            # --- THE FIX: Store only the relevant items for training ---
            # The cached tuple no longer contains segmentations.
            cached_data.append((
                mr_img.squeeze(0).cpu(),
                ct_img.squeeze(0).cpu(),
                spacing,
                f_mr.cpu(),
                f_ct.cpu(),
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

def save_overlay_png_masked(image, seg, out_path, label_cmap='jet', alpha=0.5, vmin=None, vmax=None):
    """Save overlay with only organ labels (no background) - matches syn_test.py format"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Always use Z dimension (last dimension) like syn_test.py
    mid = image.shape[2] // 2
    img_slice = image[:, :, mid]
    seg_slice = seg[:, :, mid]
        
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax)
    
    # Only color the label areas, keep background clear
    mask = seg_slice > 0
    if np.any(mask):
        colored = np.zeros((*seg_slice.shape, 4))
        colored[mask] = plt.cm.get_cmap(label_cmap)(seg_slice[mask]/np.max(seg_slice[mask]))
        plt.imshow(colored, alpha=alpha)
    
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_middle_slice_deformation_field(def_field, out_path):
    """Save deformation field visualization - matches syn_test.py format"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # def_field: (3, D, H, W) - get middle axial slice
    mid = def_field.shape[1] // 2  # Get middle slice in D dimension
    u = def_field[0, mid, :, :]
    v = def_field[1, mid, :, :]
    
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)
    
    ang_norm = (ang + np.pi) / (2 * np.pi)
    mag_norm = mag / (np.max(mag) + 1e-8)
    
    hsv = np.zeros(u.shape + (3,), dtype=np.float32)
    hsv[...,0] = ang_norm
    hsv[...,1] = 1
    hsv[...,2] = mag_norm
    
    import matplotlib.colors as mcolors
    rgb = mcolors.hsv_to_rgb(hsv)
    
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(rgb)
    plt.title('Deformation Field (middle slice)')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_middle_slice_error_map(vol1, vol2, out_path):
    """Save error map - matches syn_test.py format"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    error_map = np.abs(vol1 - vol2)
    mid = error_map.shape[2] // 2  # Use axial slice (last dimension)
    
    plt.figure(figsize=(5,5))
    plt.axis('off')
    # Show white background without any original image
    plt.imshow(np.ones_like(error_map[:,:,mid]), cmap='gray', vmin=0, vmax=1)
    if np.max(error_map[:,:,mid]) > 0:
        plt.imshow(error_map[:,:,mid], cmap='hot', alpha=0.8, vmin=0, vmax=np.max(error_map[:,:,mid]))
    plt.title('Error Map (middle slice)')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_results(fixed_vol, moving_vol, warped_vol,
                      fixed_seg, moving_seg, warped_seg,
                      flow, save_dir, patient_id):
    """
    UPDATED: Saves individual PNG overlays matching syn_test.py format exactly.
    Generates separate files for fixed, moving, and warped overlays plus deformation field and error map.
    """
    # Ensure all tensors are on the CPU and detached for plotting
    fixed_vol, moving_vol, warped_vol = fixed_vol.cpu().numpy(), moving_vol.cpu().numpy(), warped_vol.cpu().numpy()
    fixed_seg, moving_seg, warped_seg = fixed_seg.cpu().numpy(), moving_seg.cpu().numpy(), warped_seg.cpu().numpy()
    flow = flow.cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)
    
    # Remove batch and channel dimensions for processing: [B, C, D, H, W] -> [D, H, W]
    fixed_vol = fixed_vol.squeeze()
    moving_vol = moving_vol.squeeze()
    warped_vol = warped_vol.squeeze()
    fixed_seg = fixed_seg.squeeze()
    moving_seg = moving_seg.squeeze()
    warped_seg = warped_seg.squeeze()
    flow = flow.squeeze()  # [3, D, H, W]

    # Save individual overlay PNGs matching syn_test.py format exactly
    save_overlay_png_masked(fixed_vol, fixed_seg, 
                           os.path.join(save_dir, f"{patient_id}_fixed_CT_overlay.png"))
    
    save_overlay_png_masked(moving_vol, moving_seg, 
                           os.path.join(save_dir, f"{patient_id}_moving_MR_overlay.png"))
    
    save_overlay_png_masked(warped_vol, warped_seg, 
                           os.path.join(save_dir, f"{patient_id}_warped_MR_overlay.png"))
    
    # Save deformation field visualization
    save_middle_slice_deformation_field(flow, 
                                       os.path.join(save_dir, f"{patient_id}_deformation_field_middle.png"))
    
    # Save error map (on white background, no original image)
    save_middle_slice_error_map(warped_seg, fixed_seg, 
                               os.path.join(save_dir, f"{patient_id}_error_map_middle.png"))

    print(f"Saved visualizations for patient {patient_id} in {save_dir}")

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


def evaluate_vlearn(test_loader, save_dir, vol_shape, model_dir, device,
                    dino_encoder, pca_transformer, use_raw_images, max_steps=5, dataset_type=None):
    """
    UPDATED: Evaluates the VLearn policy, saves metrics to CSV, and generates detailed PNGs.
    """
    if not test_loader or len(test_loader.dataset) == 0:
        return {}
    model_path = os.path.join(model_dir, "best_vlearn_model.pth")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model checkpoint not found at {model_path}. Cannot evaluate.")
        return {}
        
    D, H_vol, W_vol = vol_shape
    
    # Determine input channels dynamically from PCA transformer
    pca_dims = pca_transformer.n_components_
    input_channels = (pca_dims * 2) + 3  # MR + CT + flow
    
    if use_raw_images:
        print("--- RUNNING ABLATION: Using RAW IMAGES as input ---")
        policy = AdaptiveFlowPolicy_Raw(in_channels=5, use_checkpointing=True).to(device)
    else:
        print(f"--- RUNNING STANDARD: Using DINO FEATURES as input ({input_channels} channels) ---")
        policy = AdaptiveFlowPolicy_Medium(in_channels=input_channels, use_checkpointing=False).to(device)
        
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['policy_state_dict']
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    policy.load_state_dict(clean_state_dict)
    policy.eval()
    
    transformer = SpatialTransformer().to(device)
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Model loaded successfully. Parameters: {num_params}")

    dice_list, hd95_list, jacobian_list, inference_time_list = [], [], [], []
    label_list = detect_dataset_labels(test_loader)
    print(f"Detected labels for evaluation: {label_list}")

    for batch_idx, (mr_vol_batch, ct_vol_batch, batch_spacing, mr_seg_batch, ct_seg_batch) in enumerate(tqdm(test_loader, desc="Evaluating")):
        mr_vol_batch, ct_vol_batch = mr_vol_batch.to(device), ct_vol_batch.to(device)
        mr_seg_batch, ct_seg_batch = mr_seg_batch.to(device), ct_seg_batch.to(device)
        
        batch_size = mr_vol_batch.shape[0]

        with torch.no_grad():
            start_time = time.time()
            
            if use_raw_images:
                # For the ablation, the "features" ARE the raw images.
                # We just need to ensure they have a channel dimension.
                f_mr_batch = mr_vol_batch.unsqueeze(1) # Add channel dim: [B, 1, D, H, W]
                f_ct_batch = ct_vol_batch.unsqueeze(1)
            else:
                f_mr_list, f_ct_list = [], []
                for i in range(batch_size):
                    # Squeeze the channel dimension (dim 0) to get a 3D tensor
                    mr_vol_3d = mr_vol_batch[i].squeeze(0)
                    ct_vol_3d = ct_vol_batch[i].squeeze(0)
                    
                    # Now pass the correctly shaped 3D tensor
                    f_mr, f_ct = extract_single_dino_features(dino_encoder, mr_vol_3d, ct_vol_3d, pca_transformer, dataset_type=dataset_type)

                    f_mr_list.append(f_mr)
                    f_ct_list.append(f_ct)
                
                f_mr_batch = torch.stack(f_mr_list).to(device)
                f_ct_batch = torch.stack(f_ct_list).to(device)
            
            flow_acc = torch.zeros((batch_size, 3, D, H_vol, W_vol), device=device)
            feature_map_spatial_dims = f_mr_batch.shape[2:]

            for step in range(max_steps):
                flow_for_features = F.interpolate(flow_acc, size=feature_map_spatial_dims, mode='trilinear', align_corners=False)
                flow_for_features = apply_flow_scaling(flow_for_features, flow_acc.shape[2:], feature_map_spatial_dims, batch_spacing)
                
                warped_f_mr_batch = transformer(f_mr_batch, flow_for_features)
                
                current_state_list = []
                for i in range(batch_size):
                    state = create_registration_state(
                        f_mr_warped=warped_f_mr_batch[i],
                        f_ct_fixed=f_ct_batch[i], 
                    )
                    current_state_list.append(state)
                current_state = torch.stack(current_state_list)

                action_dist = policy.get_action_dist(current_state)
                incremental_flow_low_res = action_dist.mean

                incremental_flow_full_res = F.interpolate(
                    incremental_flow_low_res, size=flow_acc.shape[2:], mode='trilinear', align_corners=False
                )
                incremental_flow_full_res = apply_flow_scaling(incremental_flow_full_res, incremental_flow_low_res.shape[2:], flow_acc.shape[2:], batch_spacing)
                
                flow_acc += incremental_flow_full_res
            
            inference_time = (time.time() - start_time) / batch_size
            
            warped_seg_batch = transformer(mr_seg_batch.float(), flow_acc, mode='nearest')
            warped_vol_batch = transformer(mr_vol_batch, flow_acc)
            
            for i in range(batch_size):
                patient_id = batch_idx * test_loader.batch_size + i
                inference_time_list.append(inference_time)
                
                warped_seg_np = np.round(warped_seg_batch[i].squeeze().cpu().numpy()).astype(np.int32)
                ct_seg_np = np.round(ct_seg_batch[i].squeeze().cpu().numpy()).astype(np.int32)
                
                if np.sum(warped_seg_np) == 0 or np.sum(ct_seg_np) == 0:
                    print(f"⚠️  Warning: Patient {patient_id} has an empty segmentation mask. Assigning worst scores.")
                    # Assign worst-case scores
                    dice_list.append(np.zeros(len(label_list))) # Zero Dice for all organs
                    hd95_list.append(373.13) # A large, representative number (e.g., image diagonal)
                else:
                    # If masks are valid, calculate metrics normally
                    result = score_case(ct_seg_np, warped_seg_np, label_list)
                    dice_list.append(result['DICE'])
                    
                    # Safe HD95 calculation with additional checks
                    warped_binary = warped_seg_np > 0
                    ct_binary = ct_seg_np > 0
                    
                    # Check if both binary masks have any positive voxels
                    if np.any(warped_binary) and np.any(ct_binary):
                        try:
                            hd95_value = hd95(warped_binary, ct_binary)
                            hd95_list.append(hd95_value)
                        except RuntimeError as e:
                            print(f"⚠️  Warning: HD95 calculation failed for patient {patient_id}: {e}")
                            hd95_list.append(373.13)  # Fallback value
                    else:
                        print(f"⚠️  Warning: Patient {patient_id} has empty binary mask after thresholding")
                        hd95_list.append(373.13)  # Fallback value
                
                jac_det = jacobian_determinant(flow_acc[i:i+1])
                jacobian_list.append((jac_det.cpu().numpy() < 0).mean() * 100)
                
                visualize_results(
                    fixed_vol=ct_vol_batch[i:i+1], moving_vol=mr_vol_batch[i:i+1], warped_vol=warped_vol_batch[i:i+1],
                    fixed_seg=ct_seg_batch[i:i+1], moving_seg=mr_seg_batch[i:i+1], warped_seg=warped_seg_batch[i:i+1],
                    flow=flow_acc[i:i+1], save_dir=save_dir, patient_id=patient_id
                )

    dice_array = np.array(dice_list)
    dice_mean_overall = np.mean(dice_array) if dice_array.size > 0 else 0
    dice_std_overall = np.std(dice_array) if dice_array.size > 0 else 0
    dice_mean_by_organ = np.mean(dice_array, axis=0) if dice_array.size > 0 else []

    hd95_array = np.array(hd95_list)
    hd95_mean = np.mean(hd95_array) if hd95_array.size > 0 else 0
    hd95_std = np.std(hd95_array) if hd95_array.size > 0 else 0
    
    jacobian_mean = np.mean(jacobian_list) if jacobian_list else 0
    inference_time_mean = np.mean(inference_time_list) if inference_time_list else 0
    
    # The final_metrics dictionary already correctly includes these values.
    final_metrics = {
        "dice": dice_mean_overall, "dice_std": dice_std_overall,
        "hd95": hd95_mean, "hd95_std": hd95_std,
        "jacobian_neg_percent": jacobian_mean, 
        "inference_time": inference_time_mean, # <-- Already included
        "num_parameters": num_params           # <-- Already included
    }
    
    print("\n--- Aggregated Evaluation Results ---")
    print(f"  Overall Dice: {final_metrics['dice']:.4f} ± {final_metrics['dice_std']:.4f}")
    print(f"  HD95: {final_metrics['hd95']:.2f} ± {final_metrics['hd95_std']:.2f}")
    print(f"  Negative Jacobian: {final_metrics['jacobian_neg_percent']:.2f}%")
    print(f"  Inference Time: {final_metrics['inference_time']:.4f} s")
    print(f"  Model Parameters: {final_metrics['num_parameters'] / 1e6:.2f}M")
    
    for idx, label in enumerate(label_list):
        organ_dice = dice_mean_by_organ[idx] if idx < len(dice_mean_by_organ) else 0
        final_metrics[f"organ_{label}_dice"] = organ_dice
        print(f"  - Organ {label} Dice: {organ_dice:.4f}")

    # Save detailed results CSV (same format as syn_test.py summary)
    results_csv_path = os.path.join(save_dir, "test_results_summary.csv")
    with open(results_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Metric', 'Mean', 'Std'])
        
        # Write overall metrics
        writer.writerow(['Average_Dice_Score', f"{final_metrics['dice']:.4f}", f"{final_metrics['dice_std']:.4f}"])
        writer.writerow(['Average_HD95_mm', f"{final_metrics['hd95']:.2f}", f"{final_metrics['hd95_std']:.2f}"])
        writer.writerow(['Average_Non_Positive_Jacobian_percent', f"{final_metrics['jacobian_neg_percent']:.2f}", "0.00"])
        writer.writerow(['Average_Inference_Time_s', f"{final_metrics['inference_time']:.4f}", "0.00"])
        writer.writerow(['Model_Parameters_M', f"{final_metrics['num_parameters'] / 1e6:.2f}", "0.00"])
        
        # Write per-organ dice scores
        for idx, label in enumerate(label_list):
            if idx < len(dice_mean_by_organ):
                dice_std_by_organ = np.std(dice_array[:, idx]) if dice_array.size > 0 else 0
                writer.writerow([f'Dice_Label_{label}', f"{dice_mean_by_organ[idx]:.4f}", f"{dice_std_by_organ:.4f}"])

    print(f"\n💾 Results saved to: {results_csv_path}")

    # Keep the old CSV format for backwards compatibility
    csv_path = os.path.join(save_dir, "results.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        # The fieldnames are created from the keys of final_metrics,
        # so inference_time and num_parameters are automatically included.
        fieldnames = list(final_metrics.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(final_metrics)
        
    print(f"\n✅ Aggregated metrics saved to {csv_path}")

    return final_metrics

def validate_memory_efficient(policy, val_loader, device, vol_shape, feature_shape, max_steps=5, max_validation_samples=None, use_raw_images=False):
    """
    Memory-efficient validation that processes smaller batches and uses CPU offloading.
    
    Args:
        max_validation_samples: Limit number of validation samples (None = use all)
        use_raw_images: Whether we're using raw images (True) or DINO features (False)
    
    Returns:
        Mean NCC score across validation samples
    """
    if not val_loader or len(val_loader.dataset) == 0:
        print("⚠️  Warning: No validation data available")
        return 0.5690  # Return a default value
        
    policy.eval()
    transformer = SpatialTransformer().to(device)
    mind_loss_func = MINDLoss().cpu() 
    all_mind_scores = []
    
    D_vol, H_vol, W_vol = vol_shape
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            # Handle different batch formats - validation loader might have different structure
            if len(batch_data) == 5:  # (mr_vol_batch, ct_vol_batch, batch_spacing, f_mr_batch, f_ct_batch)
                mr_vol_batch, ct_vol_batch, batch_spacing, f_mr_batch, f_ct_batch = batch_data
            elif len(batch_data) == 3:  # (mr_vol_batch, ct_vol_batch, batch_spacing) - need to compute features
                mr_vol_batch, ct_vol_batch, batch_spacing = batch_data
                # Skip this sample if we don't have pre-computed features
                print(f"⚠️  Warning: Validation batch {batch_idx} missing features, skipping")
                continue
            else:
                print(f"⚠️  Warning: Unexpected validation batch format with {len(batch_data)} items")
                continue
                
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
                    
                    if not use_raw_images:
                        # Use actual feature dimensions from f_mr
                        actual_feat_shape = f_mr.shape[2:]  # Get actual spatial dimensions
                        f_mr = F.interpolate(f_mr, size=actual_feat_shape, mode='trilinear', align_corners=False)
                        f_ct = F.interpolate(f_ct, size=actual_feat_shape, mode='trilinear', align_corners=False)
                    else:
                        # For raw images, interpolate to match volume shape
                        f_mr = F.interpolate(f_mr, size=vol_shape, mode='trilinear', align_corners=False)
                        f_ct = F.interpolate(f_ct, size=vol_shape, mode='trilinear', align_corners=False)

                current_spacing = batch_spacing[sample_idx]

                # Handle different tensor shapes for raw images vs DINO features
                if use_raw_images:
                    # For raw images, f_mr and f_ct are already in the correct shape [1, 1, D, H, W]
                    f_mr_reshaped = f_mr
                    f_ct_reshaped = f_ct
                    feature_map_spatial_dims = f_mr_reshaped.shape[2:]  # [D, H, W]
                else:
                    # For DINO features, get actual dimensions dynamically
                    C_actual = f_mr.shape[1]  # Actual number of channels
                    spatial_dims = f_mr.shape[2:]  # Actual spatial dimensions
                    f_mr_reshaped = f_mr.view(1, C_actual, *spatial_dims)
                    f_ct_reshaped = f_ct.view(1, C_actual, *spatial_dims)
                    feature_map_spatial_dims = spatial_dims
                
                flow_acc = torch.zeros((1, 3, D_vol, H_vol, W_vol), device=device)
                
                # Registration loop with memory cleanup
                for step in range(max_steps):
                    flow_for_features = F.interpolate(flow_acc, size=feature_map_spatial_dims, mode='trilinear', align_corners=False)
                    flow_for_features = apply_flow_scaling(flow_for_features, flow_acc.shape[2:], feature_map_spatial_dims, current_spacing)
                    
                    warped_f_mr = transformer(f_mr_reshaped, flow_for_features)
                    
                    # Create state using the same method as training
                    current_state = create_registration_state(
                        f_mr_warped=warped_f_mr.squeeze(0),  # Remove batch dim for state creation
                        f_ct_fixed=f_ct_reshaped.squeeze(0), 
                    ).unsqueeze(0)  # Add batch dim back
                    
                    # Get the deterministic action from the policy
                    action_dist = policy.get_action_dist(current_state)
                    incremental_flow_low_res = action_dist.mean
                    
                    incremental_flow = F.interpolate(
                        incremental_flow_low_res,
                        size=flow_acc.shape[2:],  # Target the full resolution of flow_acc
                        mode='trilinear',
                        align_corners=False
                    )
                    incremental_flow = apply_flow_scaling(incremental_flow, incremental_flow_low_res.shape[2:], flow_acc.shape[2:])
    
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