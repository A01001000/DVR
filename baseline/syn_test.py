import os
import glob
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"  

import ants
import numpy as np
import argparse
import csv
import time
from scipy.spatial.distance import directed_hausdorff
from medpy.metric.binary import hd95

def compute_dice_score(pred_seg, true_seg, num_classes=None):
    """Compute Dice score for each class and overall average."""
    pred_arr = pred_seg.numpy().astype(np.uint8)
    true_arr = true_seg.numpy().astype(np.uint8)
    
    if num_classes is None:
        num_classes = max(np.max(pred_arr), np.max(true_arr)) + 1
    
    dice_scores = []
    for class_id in range(1, num_classes):  # Skip background (class 0)
        pred_mask = (pred_arr == class_id).astype(np.float32)
        true_mask = (true_arr == class_id).astype(np.float32)
        
        intersection = np.sum(pred_mask * true_mask)
        total = np.sum(pred_mask) + np.sum(true_mask)
        
        if total == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection) / total
        dice_scores.append(dice)
    
    return dice_scores, np.mean(dice_scores) if dice_scores else 0.0


def compute_hausdorff_95(pred_seg, true_seg, spacing=None):
    """Compute 95th percentile Hausdorff distance (surface-based)."""
    pred_arr = pred_seg.numpy().astype(np.uint8)
    true_arr = true_seg.numpy().astype(np.uint8)
    if spacing is None:
        spacing = pred_seg.spacing
    from scipy import ndimage
    pred_edges = ndimage.binary_erosion(pred_arr > 0) ^ (pred_arr > 0)
    true_edges = ndimage.binary_erosion(true_arr > 0) ^ (true_arr > 0)
    pred_points = np.argwhere(pred_edges)
    true_points = np.argwhere(true_edges)
    if len(pred_points) == 0 or len(true_points) == 0:
        return np.nan
    pred_points = pred_points * np.array(spacing)
    true_points = true_points * np.array(spacing)
    from scipy.spatial.distance import cdist
    dists = cdist(pred_points, true_points)
    if dists.size == 0:
        return np.nan
    hd95_value = np.percentile(np.hstack([dists.min(axis=1), dists.min(axis=0)]), 95)
    return hd95_value


def compute_jacobian_determinant(deformation_field):
    """Compute negative Jacobian determinant percentage."""
    def_arr = deformation_field.numpy()
    
    # Create coordinate grid
    shape = def_arr.shape[:-1]  # Remove the last dimension (vector components)
    coords = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    
    # Add deformation to coordinates
    for i in range(len(coords)):
        coords[i] = coords[i] + def_arr[..., i]
    
    # Compute gradients
    jac_det = np.ones(shape)
    if len(shape) == 3:  # 3D case
        dx_dx = np.gradient(coords[0], axis=0)
        dx_dy = np.gradient(coords[0], axis=1)
        dx_dz = np.gradient(coords[0], axis=2)
        
        dy_dx = np.gradient(coords[1], axis=0)
        dy_dy = np.gradient(coords[1], axis=1)
        dy_dz = np.gradient(coords[1], axis=2)
        
        dz_dx = np.gradient(coords[2], axis=0)
        dz_dy = np.gradient(coords[2], axis=1)
        dz_dz = np.gradient(coords[2], axis=2)
        
        # Compute determinant
        jac_det = (dx_dx * (dy_dy * dz_dz - dy_dz * dz_dy) -
                   dx_dy * (dy_dx * dz_dz - dy_dz * dz_dx) +
                   dx_dz * (dy_dx * dz_dy - dy_dy * dz_dx))
    
    negative_jac_percentage = np.sum(jac_det <= 0) / jac_det.size * 100
    return negative_jac_percentage


# --- Helper functions for saving images ---
import matplotlib.pyplot as plt
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

def save_overlay_png_masked(image, seg, out_path, label_cmap='jet', alpha=0.5, vmin=None, vmax=None):
    mid = image.shape[2] // 2
    img_slice = image[:,:,mid]
    seg_slice = seg[:,:,mid]
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


# --- Robust per-label Dice and HD95 functions (from DINO-Reg style) ---
def compute_dice_coefficient(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    size1 = np.sum(mask1)
    size2 = np.sum(mask2)
    return (2.0 * intersection + 1e-5) / (size1 + size2 + 1e-5)

def compute_label_wise_dice(seg1, seg2, labels):
    dice_results = []
    for label in labels:
        seg1_label = seg1 == label
        seg2_label = seg2 == label
        dice = compute_dice_coefficient(seg1_label, seg2_label)
        dice_results.append(dice)
    return dice_results

def compute_95_hausdorff_distance(seg1, seg2):
    u_indices = np.array(np.where(seg1)).T
    v_indices = np.array(np.where(seg2)).T
    if u_indices.size == 0 or v_indices.size == 0:
        return np.nan
    from scipy.spatial.distance import cdist
    distances = cdist(u_indices, v_indices, 'euclidean')
    sorted_distances = np.sort(distances, axis=None)
    hd_95 = np.percentile(sorted_distances, 95)
    return hd_95

def compute_label_wise_95hd(seg1, seg2, labels):
    hd95_results = []
    for label in labels:
        seg1_label = seg1 == label
        seg2_label = seg2 == label
        hd95 = compute_95_hausdorff_distance(seg1_label, seg2_label)
        hd95_results.append(hd95)
    return hd95_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Name of dataset (CHAOS, L2R, MRXFDG)")
    parser.add_argument("--csv_dir", type=str, default=None, help="Path to csv folder")
    parser.add_argument("--label_dir", type=str, default=None, help="Path to labels folder aka test folder")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to output folder")
    parser.add_argument("--num_labels", type=int, default=None, help="Number of organ labels (excluding background)")
    parser.add_argument("--skip_registration", action="store_true", help="Skip registration and use existing warped images/deformation fields from output_dir")
    args = parser.parse_args()


    labels_root = args.label_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize CSV results list
    csv_results = []
    csv_headers = ['patient_id', 'avg_dice', 'hd95', 'neg_jac_percent', 'inference_time', 'num_parameters']
    
    # Determine organ classes based on dataset
    organ_names = []
    if args.data == "CHAOS":
        organ_names = ['liver', 'right_kidney', 'left_kidney', 'spleen']
    elif args.data == "L2R":
        organ_names = ['liver']
    elif args.data == "MRXFDG":
        organ_names = ['liver', 'kidneys', 'spleen']
    
    # If num_labels is provided, override organ_names
    if args.num_labels is not None:
        organ_names = [f'label_{i}' for i in range(1, args.num_labels+1)]
    
    # Add organ-specific dice columns to headers
    for organ in organ_names:
        csv_headers.append(f'dice_{organ}')

    # Only include directories that do not start with a dot (skip .DS_Store, etc)
    patient_ids = sorted([d for d in os.listdir(labels_root) if os.path.isdir(os.path.join(labels_root, d)) and not d.startswith('.')])

    # For accumulating pre-registration metrics
    pre_dice_list = []
    pre_hd95_list = []
    pre_dice_per_organ_list = []
    pre_patient_ids = []

    # For accumulating post-registration metrics for summary
    post_dice_list = []
    post_hd95_list = []
    post_dice_per_organ_list = []
    post_jac_list = []
    post_time_list = []
    post_num_param_list = []

    for pid in patient_ids:
        patient_dir = os.path.join(labels_root, pid)
        outpid_dir = os.path.join(output_dir, pid)
        if os.path.isdir(patient_dir):
            mr_folder = os.path.join(patient_dir, "MR")
            ct_folder = os.path.join(patient_dir, "CT")
            mr_label_folder = os.path.join(patient_dir, "MR_seg")
            ct_label_folder = os.path.join(patient_dir, "CT_seg")
        
            # Only include .nii or .nii.gz files
            mr_files = [f for f in glob.glob(os.path.join(mr_folder, "*.nii*")) if f.endswith('.nii') or f.endswith('.nii.gz')]
            ct_files = [f for f in glob.glob(os.path.join(ct_folder, "*.nii*")) if f.endswith('.nii') or f.endswith('.nii.gz')]
            mr_label_files = [f for f in glob.glob(os.path.join(mr_label_folder, "*.nii*")) if f.endswith('.nii') or f.endswith('.nii.gz')]
            ct_label_files = [f for f in glob.glob(os.path.join(ct_label_folder, "*.nii*")) if f.endswith('.nii') or f.endswith('.nii.gz')]

            mr_path = mr_files[0] if mr_files else None
            ct_path = ct_files[0] if ct_files else None
            mr_label = mr_label_files[0] if mr_label_files else None
            ct_label = ct_label_files[0] if ct_label_files else None

            if not all([mr_path, ct_path, mr_label, ct_label]):
                print(f"Missing files for patient {pid}, skipping...")
                continue

            out_dir = os.path.join(outpid_dir, "MR_SyN")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{args.data}_{pid}_MR_SyN.nii.gz")
            deformation_path = os.path.join(out_dir, f"{args.data}_{pid}_deformation_field.nii.gz")

            print(f"Registering patient {pid}...")

            fixed = ants.image_read(ct_path)
            moving = ants.image_read(mr_path)
            moving_seg = ants.image_read(mr_label)
            fixed_seg = ants.image_read(ct_label)

            # Debug: print sum and unique values after loading
            print(f"[DEBUG] Patient {pid} loaded MR_seg sum: {moving_seg.numpy().sum()}, unique: {np.unique(moving_seg.numpy())}")
            print(f"[DEBUG] Patient {pid} loaded CT_seg sum: {fixed_seg.numpy().sum()}, unique: {np.unique(fixed_seg.numpy())}")

            # --- Align MR image to CT's physical space for registration only ---
            moving = ants.resample_image_to_target(moving, fixed)
            # Do NOT resample moving_seg or fixed_seg for metric computation

            # Debug: print sum and unique values after loading (segmentations are not resampled)
            print(f"[DEBUG] Patient {pid} loaded MR_seg sum: {moving_seg.numpy().sum()}, unique: {np.unique(moving_seg.numpy())}")
            print(f"[DEBUG] Patient {pid} loaded CT_seg sum: {fixed_seg.numpy().sum()}, unique: {np.unique(fixed_seg.numpy())}")

            # Debug: print image headers before resampling
            print(f"[DEBUG] Patient {pid} MR_seg origin: {moving_seg.origin}, spacing: {moving_seg.spacing}, direction: {moving_seg.direction}")
            print(f"[DEBUG] Patient {pid} CT origin: {fixed.origin}, spacing: {fixed.spacing}, direction: {fixed.direction}")

            # Debug: print shapes before resampling
            print(f"[DEBUG] Patient {pid} MR_seg shape: {moving_seg.shape}")
            print(f"[DEBUG] Patient {pid} CT shape: {fixed.shape}")
            # Debug: print bounding box of nonzero region in MR_seg
            nz = np.argwhere(moving_seg.numpy() > 0)
            if nz.size > 0:
                print(f"[DEBUG] Patient {pid} MR_seg bbox: min={nz.min(axis=0)}, max={nz.max(axis=0)}")
            else:
                print(f"[DEBUG] Patient {pid} MR_seg bbox: EMPTY")

            # Debug: print MR_seg nonzero values and dtype before resampling
            mrseg_np = moving_seg.numpy()
            print(f"[DEBUG] Patient {pid} MR_seg dtype: {mrseg_np.dtype}")
            nonzero_vals = np.unique(mrseg_np[mrseg_np != 0])
            print(f"[DEBUG] Patient {pid} MR_seg nonzero values: {nonzero_vals}")

            # Skip patient if either segmentation is empty
            if np.sum(moving_seg.numpy()) == 0 or np.sum(fixed_seg.numpy()) == 0:
                print(f"[WARNING] Patient {pid} skipped: segmentation is empty.")
                continue

            # If needed, re-normalize just before registration
            fixed = ants.iMath(fixed, "Normalize")
            moving = ants.iMath(moving, "Normalize")

            # --- Compute pre-registration metrics (Dice, HD95) and overlays as usual ---
            # Save MR/CT with segmentation overlay (middle slice)
            import matplotlib.pyplot as plt
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

            # --- Save overlays for CT, MR, and warped MR with label color only on organ areas ---
            def save_overlay_png_masked(image, seg, out_path, label_cmap='jet', alpha=0.5, vmin=None, vmax=None):
                mid = image.shape[2] // 2
                img_slice = image[:,:,mid]
                seg_slice = seg[:,:,mid]
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

            save_overlay_png_masked(moving.numpy(), moving_seg.numpy(), os.path.join(out_dir, f"{args.data}_{pid}_MR_overlay_pre.png"))
            save_overlay_png_masked(fixed.numpy(), fixed_seg.numpy(), os.path.join(out_dir, f"{args.data}_{pid}_CT_overlay_pre.png"))

            if args.skip_registration:
                # Load warped image and deformation field if they exist
                warped_seg_path = os.path.join(out_dir, f"{args.data}_{pid}_MR_SyN.nii.gz")
                warped_img_path = os.path.join(out_dir, f"{args.data}_{pid}_MR_warped.nii.gz")
                deformation_path = os.path.join(out_dir, f"{args.data}_{pid}_deformation_field.nii.gz")
                if os.path.exists(warped_seg_path) and os.path.exists(warped_img_path):
                    warped_moving_seg = ants.image_read(warped_seg_path)
                    warped_moving_img = ants.image_read(warped_img_path)
                    # Save warped MR middle slice
                    save_middle_slice_png(warped_moving_img.numpy(), os.path.join(out_dir, f"{args.data}_{pid}_MR_warped_middle.png"))
                    # Save overlay for warped MR (warped MR image as background, warped labels as overlay)
                    save_overlay_png_masked(warped_moving_img.numpy(), warped_moving_seg.numpy(), os.path.join(out_dir, f"{args.data}_{pid}_MR_warped_overlay.png"))
                    # --- Save error map of the middle slice after warping (on white background) ---
                    error_map = np.abs(warped_moving_seg.numpy() - fixed_seg.numpy())
                    mid = error_map.shape[2] // 2
                    plt.figure(figsize=(5,5))
                    plt.axis('off')
                    plt.imshow(np.ones_like(error_map[:,:,mid]), cmap='gray', vmin=0, vmax=1)
                    if np.max(error_map[:,:,mid]) > 0:
                        plt.imshow(error_map[:,:,mid], cmap='hot', alpha=0.8, vmin=0, vmax=np.max(error_map[:,:,mid]))
                    plt.title('Error Map (middle slice)')
                    plt.savefig(os.path.join(out_dir, f"{args.data}_{pid}_error_map_middle.png"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    # Compute post-registration metrics
                    try:
                        dice_per_organ, avg_dice = compute_dice_score(warped_moving_seg, fixed_seg, num_classes=len(organ_names)+1)
                        hd95_value = compute_hausdorff_95(warped_moving_seg, fixed_seg)
                        # Try to load deformation field for Jacobian
                        if os.path.exists(deformation_path):
                            deformation_field = ants.image_read(deformation_path)
                            neg_jac_percent = compute_jacobian_determinant(deformation_field)
                        else:
                            neg_jac_percent = np.nan
                        num_parameters = 0
                        post_dice_list.append(avg_dice)
                        post_hd95_list.append(hd95_value)
                        post_dice_per_organ_list.append(dice_per_organ)
                        post_jac_list.append(neg_jac_percent)
                        post_time_list.append(np.nan)
                        post_num_param_list.append(num_parameters)
                        # Prepare CSV row
                        csv_row = [pid, avg_dice, hd95_value, neg_jac_percent, np.nan, num_parameters]
                        for i, dice in enumerate(dice_per_organ):
                            csv_row.append(dice)
                        while len(csv_row) < len(csv_headers):
                            csv_row.append(0.0)
                        csv_results.append(csv_row)
                        print(f"Patient {pid} - Avg Dice: {avg_dice:.4f}, HD95: {hd95_value:.2f}, Neg Jac: {neg_jac_percent}, Time: n/a")
                    except Exception as e:
                        print(f"Evaluation failed for {pid}: {e}")
                        csv_row = [pid] + [float('nan')] * (len(csv_headers) - 1)
                        csv_results.append(csv_row)
                else:
                    print(f"No warped image or segmentation found for {pid}, skipping post-registration metrics.")
                continue  # Skip registration

            try:
                # Start timing
                start_time = time.time()
                
                registration = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')
                
                # End timing
                inference_time = time.time() - start_time
                
            except Exception as e:
                print(f"Registration failed for {pid}: {e}")
                continue

            warped_moving_seg = ants.apply_transforms(
                fixed=fixed,
                moving=moving_seg,
                transformlist=registration['fwdtransforms'],
                interpolator='nearestNeighbor'
            )

            # Save the warped segmentation
            ants.image_write(warped_moving_seg, out_path)
            print(f"Saved: {out_path}")
            
            # Save deformation field
            if 'fwdtransforms' in registration and len(registration['fwdtransforms']) > 0:
                for transform_file in registration['fwdtransforms']:
                    if transform_file.endswith('.nii.gz'):
                        deformation_field = ants.image_read(transform_file)
                        ants.image_write(deformation_field, deformation_path)
                        print(f"Saved deformation field: {deformation_path}")
                        break

            # --- After registration, also warp and save the MR image itself
            warped_moving_img = ants.apply_transforms(
                fixed=fixed,
                moving=moving,
                transformlist=registration['fwdtransforms'],
                interpolator='linear'
            )
            warped_moving_img_path = os.path.join(out_dir, f"{args.data}_{pid}_MR_warped.nii.gz")
            ants.image_write(warped_moving_img, warped_moving_img_path)
            print(f"Saved: {warped_moving_img_path}")

            # Save original MR, CT, warped MR middle slices
            save_middle_slice_png(moving.numpy(), os.path.join(out_dir, f"{args.data}_{pid}_MR_middle.png"))
            save_middle_slice_png(fixed.numpy(), os.path.join(out_dir, f"{args.data}_{pid}_CT_middle.png"))
            save_middle_slice_png(warped_moving_img.numpy(), os.path.join(out_dir, f"{args.data}_{pid}_MR_warped_middle.png"))

            # --- Visualize deformation field (middle slice, color-coded) ---
            # deformation_field: shape (X, Y, Z, 3) or (X, Y, Z, 2)
            if 'deformation_field' in locals():
                def_field = deformation_field.numpy()
                if def_field.shape[-1] == 3:
                    # Take the middle slice in Z
                    mid = def_field.shape[2] // 2
                    u = def_field[:,:,mid,0]
                    v = def_field[:,:,mid,1]
                    # Compute magnitude and angle
                    mag = np.sqrt(u**2 + v**2)
                    ang = np.arctan2(v, u)
                    # Normalize for HSV
                    ang_norm = (ang + np.pi) / (2 * np.pi)  # [0,1]
                    mag_norm = mag / (np.max(mag) + 1e-8)   # [0,1]
                    hsv = np.zeros(u.shape + (3,), dtype=np.float32)
                    hsv[...,0] = ang_norm  # Hue: direction
                    hsv[...,1] = 1         # Saturation
                    hsv[...,2] = mag_norm  # Value: magnitude
                    import matplotlib.colors as mcolors
                    rgb = mcolors.hsv_to_rgb(hsv)
                    plt.figure(figsize=(5,5))
                    plt.axis('off')
                    plt.imshow(rgb)
                    plt.title('Deformation Field (middle slice)')
                    plt.savefig(os.path.join(out_dir, f"{args.data}_{pid}_deformation_field_middle.png"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                else:
                    # If not 3D vector, fallback to quiver plot for 2D
                    mid = def_field.shape[2] // 2
                    u = def_field[:,:,mid,0]
                    v = def_field[:,:,mid,1]
                    plt.figure(figsize=(5,5))
                    plt.axis('off')
                    plt.quiver(u, v)
                    plt.title('Deformation Field (middle slice)')
                    plt.savefig(os.path.join(out_dir, f"{args.data}_{pid}_deformation_field_middle.png"), bbox_inches='tight', pad_inches=0)
                    plt.close()
            
            # --- Compute and save metrics/PNGs for original (pre-registration) images ---
            # Save MR/CT with segmentation overlay (middle slice)
            import matplotlib.pyplot as plt
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

            # --- Save overlays for CT, MR, and warped MR with label color only on organ areas ---
            def save_overlay_png_masked(image, seg, out_path, label_cmap='jet', alpha=0.5, vmin=None, vmax=None):
                mid = image.shape[2] // 2
                img_slice = image[:,:,mid]
                seg_slice = seg[:,:,mid]
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

            save_overlay_png_masked(moving.numpy(), moving_seg.numpy(), os.path.join(out_dir, f"{args.data}_{pid}_MR_overlay_pre.png"))
            save_overlay_png_masked(fixed.numpy(), fixed_seg.numpy(), os.path.join(out_dir, f"{args.data}_{pid}_CT_overlay_pre.png"))
            # After registration, save overlay for warped MR (warped MR image as background, warped labels as overlay)
            save_overlay_png_masked(warped_moving_img.numpy(), warped_moving_seg.numpy(), os.path.join(out_dir, f"{args.data}_{pid}_MR_warped_overlay.png"))

            # --- Save error map of the middle slice after warping (on white background) ---
            error_map = np.abs(warped_moving_seg.numpy() - fixed_seg.numpy())
            mid = error_map.shape[2] // 2
            plt.figure(figsize=(5,5))
            plt.axis('off')
            # Show white background
            plt.imshow(np.ones_like(error_map[:,:,mid]), cmap='gray', vmin=0, vmax=1)
            # Overlay error map
            if np.max(error_map[:,:,mid]) > 0:
                plt.imshow(error_map[:,:,mid], cmap='hot', alpha=0.8, vmin=0, vmax=np.max(error_map[:,:,mid]))
            plt.title('Error Map (middle slice)')
            plt.savefig(os.path.join(out_dir, f"{args.data}_{pid}_error_map_middle.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

            # Compute pre-registration metrics (Dice, HD95)
            try:
                # Use all unique nonzero labels for per-label metrics
                label_set = sorted([int(x) for x in np.unique(moving_seg.numpy()) if x != 0])
                dice_per_organ_pre = compute_label_wise_dice(moving_seg.numpy(), fixed_seg.numpy(), label_set)
                avg_dice_pre = np.mean(dice_per_organ_pre)
                hd95_value_pre = np.mean(compute_label_wise_95hd(moving_seg.numpy(), fixed_seg.numpy(), label_set))
                pre_dice_list.append(avg_dice_pre)
                pre_hd95_list.append(hd95_value_pre)
                pre_dice_per_organ_list.append(dice_per_organ_pre)
                pre_patient_ids.append(pid)
            except Exception as e:
                pre_dice_list.append(np.nan)
                pre_hd95_list.append(np.nan)
                pre_dice_per_organ_list.append([np.nan]*len(organ_names))
                pre_patient_ids.append(pid)

            # Compute evaluation metrics
            try:
                label_set = sorted([int(x) for x in np.unique(moving_seg.numpy()) if x != 0])
                dice_per_organ = compute_label_wise_dice(warped_moving_seg.numpy(), fixed_seg.numpy(), label_set)
                avg_dice = np.mean(dice_per_organ)
                hd95_value = np.mean(compute_label_wise_95hd(warped_moving_seg.numpy(), fixed_seg.numpy(), label_set))
                if 'fwdtransforms' in registration:
                    neg_jac_percent = 0.0
                    for transform_file in registration['fwdtransforms']:
                        if transform_file.endswith('.nii.gz'):
                            deformation_field = ants.image_read(transform_file)
                            neg_jac_percent = compute_jacobian_determinant(deformation_field)
                            break
                else:
                    neg_jac_percent = 0.0
                num_parameters = 0
                post_dice_list.append(avg_dice)
                post_hd95_list.append(hd95_value)
                post_dice_per_organ_list.append(dice_per_organ)
                post_jac_list.append(neg_jac_percent)
                post_time_list.append(inference_time)
                post_num_param_list.append(num_parameters)
                # Prepare CSV row
                csv_row = [pid, avg_dice, hd95_value, neg_jac_percent, inference_time, num_parameters]
                for i, dice in enumerate(dice_per_organ):
                    csv_row.append(dice)
                while len(csv_row) < len(csv_headers):
                    csv_row.append(0.0)
                csv_results.append(csv_row)
                print(f"Patient {pid} - Avg Dice: {avg_dice:.4f}, HD95: {hd95_value:.2f}, "
                      f"Neg Jac: {neg_jac_percent:.2f}%, Time: {inference_time:.2f}s")
            except Exception as e:
                print(f"Evaluation failed for {pid}: {e}")
                csv_row = [pid] + [float('nan')] * (len(csv_headers) - 1)
                csv_results.append(csv_row)

    # Save CSV results
    csv_output_path = os.path.join(args.csv_dir, f"{args.data}_SyN_evaluation_results.csv")
    with open(csv_output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
        writer.writerows(csv_results)
    
    print(f"Evaluation results saved to: {csv_output_path}")
    
    # Print summary statistics
    if csv_results:
        valid_results = [row for row in csv_results if not any(np.isnan(x) if isinstance(x, (int, float)) else False for x in row[1:])]
        if valid_results:
            avg_dice_all = np.mean([row[1] for row in valid_results])
            avg_hd95_all = np.mean([row[2] for row in valid_results])
            avg_neg_jac_all = np.mean([row[3] for row in valid_results])
            avg_time_all = np.mean([row[4] for row in valid_results])
            
            print(f"\nSummary Statistics:")
            print(f"Average Dice Score: {avg_dice_all:.4f}")
            print(f"Average HD95: {avg_hd95_all:.2f}")
            print(f"Average Negative Jacobian %: {avg_neg_jac_all:.2f}")
            print(f"Average Inference Time: {avg_time_all:.2f}s")
            print(f"Total processed patients: {len(valid_results)}")
    
    # Save pre-registration metrics as a single CSV (mean/std across all patients)
    if pre_dice_list:
        pre_dice_array = np.array(pre_dice_list)
        pre_hd95_array = np.array(pre_hd95_list)
        pre_dice_per_organ_array = np.array(pre_dice_per_organ_list)
        pre_csv_headers = ['avg_dice', 'hd95'] + [f'dice_{organ}' for organ in organ_names]
        pre_csv_path = os.path.join(output_dir, f"{args.data}_pre_registration_metrics.csv")
        with open(pre_csv_path, 'w', newline='') as pre_csvfile:
            pre_writer = csv.writer(pre_csvfile)
            pre_writer.writerow(['Metric', 'Mean', 'Std'])
            pre_writer.writerow(['avg_dice', np.nanmean(pre_dice_array), np.nanstd(pre_dice_array)])
            pre_writer.writerow(['hd95', np.nanmean(pre_hd95_array), np.nanstd(pre_hd95_array)])
            for i, organ in enumerate(organ_names):
                pre_writer.writerow([f'dice_{organ}', np.nanmean(pre_dice_per_organ_array[:,i]), np.nanstd(pre_dice_per_organ_array[:,i])])
        print(f"Pre-registration metrics saved to: {pre_csv_path}")
    
    # Save summary CSV for post-registration metrics
    if post_dice_list:
        post_dice_array = np.array(post_dice_list)
        post_hd95_array = np.array(post_hd95_list)
        post_dice_per_organ_array = np.array(post_dice_per_organ_list)
        post_jac_array = np.array(post_jac_list)
        post_time_array = np.array(post_time_list)
        post_num_param_array = np.array(post_num_param_list)
        summary_csv_path = os.path.join(output_dir, f"{args.data}_SyN_evaluation_summary.csv")
        with open(summary_csv_path, 'w', newline='') as summary_csvfile:
            summary_writer = csv.writer(summary_csvfile)
            summary_writer.writerow(['Metric', 'Mean', 'Std'])
            summary_writer.writerow(['avg_dice', np.nanmean(post_dice_array), np.nanstd(post_dice_array)])
            summary_writer.writerow(['hd95', np.nanmean(post_hd95_array), np.nanstd(post_hd95_array)])
            summary_writer.writerow(['non_pos_jac', np.nanmean(post_jac_array), np.nanstd(post_jac_array)])
            summary_writer.writerow(['inference_time', np.nanmean(post_time_array), np.nanstd(post_time_array)])
            summary_writer.writerow(['num_parameters', np.nanmean(post_num_param_array), np.nanstd(post_num_param_array)])
            for i, organ in enumerate(organ_names):
                summary_writer.writerow([f'dice_{organ}', np.nanmean(post_dice_per_organ_array[:,i]), np.nanstd(post_dice_per_organ_array[:,i])])
        print(f"Post-registration summary metrics saved to: {summary_csv_path}")