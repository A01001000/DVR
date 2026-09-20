import joblib
import torch
import os
import traceback  # Add this import
from glob import glob
import argparse
from tqdm import tqdm
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
import torchio as tio
import nibabel as nib
from tqdm import tqdm
import argparse
import torch.nn.functional as F
import torch
import numpy as np
import torchvision.utils as vutils
from PIL import Image

from dvr2.utils import get_augmentation_transform
import traceback

def fit_pca_on_subset(dino_encoder, path_pairs, out_dim=64, samples_per_volume=100, target_size=(128, 192, 192), dataset_type=None, feature_type="hybrid"):
    """
    FIXED: Fits PCA on the CORRECT features based on dataset type and feature type.
    For brain: 
      - feature_type="hybrid": fits on hybrid L-F1-S3+DINO features (96D) -> 128D PCA
      - feature_type="standard": fits on standard DINO features (256D) -> 128D PCA
    For others: fits on standard DINO features (256D) -> 64D PCA
    """
    # Adjust PCA dimensions based on dataset type and feature type
    if dataset_type and dataset_type.lower() == "brain":
        if feature_type == "hybrid":
            # CRITICAL: For hybrid features, use PCA dimensions that match the training model
            # The training used 96D hybrid features -> 96D PCA (not 128D) to get 192 input channels
            out_dim = 96   # Keep hybrid features at 96D to match trained model dimensions
            print(f"Using {out_dim} PCA dimensions for HYBRID brain features (L-F1-S3 + DINO)")
            print("Fitting PCA on 96D hybrid features (32D L-F1-S3 + 64D DINO) -> 96D output")
        else:  # standard DINO
            out_dim = 128   # Use 128 for brain standard DINO 
            print(f"Using {out_dim} PCA dimensions for standard DINO brain features")
            print("Fitting PCA on standard 256D DINO features...")
    elif dataset_type and dataset_type.lower() == "abdomen":
        out_dim = 64   # Standard dimensions for abdomen
        print(f"Using {out_dim} PCA dimensions for abdomen dataset")
        print("Fitting PCA on standard DINO features...")
    
    print("Collecting a subset of FOREGROUND features to fit PCA...")
    all_features = []
    dino_encoder.proj_head.eval()

    for mr_path, ct_path in tqdm(path_pairs, desc="Collecting PCA Samples"):
        mr_vol_full = torch.tensor(nib.load(mr_path).get_fdata(), dtype=torch.float32)
        ct_vol_full = torch.tensor(nib.load(ct_path).get_fdata(), dtype=torch.float32)

        with torch.no_grad():
            mr_vol_downsampled = F.interpolate(mr_vol_full.unsqueeze(0).unsqueeze(0), size=target_size, mode='trilinear').squeeze()
            ct_vol_downsampled = F.interpolate(ct_vol_full.unsqueeze(0).unsqueeze(0), size=target_size, mode='trilinear').squeeze()
            
            # FIXED: Extract features using the SAME method as during training/testing
            if dataset_type and dataset_type.lower() == "brain":
                if feature_type == "hybrid":
                    # Create hybrid features (L-F1-S3 + DINO) - same as in extract_single_dino_features
                    try:
                        # Step 1: Extract L-F1-S3 features (32D)
                        l_f1_s3_mr, l_f1_s3_ct = extract_brain_guided_features(dino_encoder, mr_vol_downsampled, ct_vol_downsampled, dataset_type)
                        
                        # Step 2: Extract standard DINO features for additional representation
                        dino_mr = dino_encoder.encode_volume(mr_vol_downsampled, use_proj=True, dataset_type=None)  # Force standard DINO
                        dino_ct = dino_encoder.encode_volume(ct_vol_downsampled, use_proj=True, dataset_type=None)
                        
                        # Reshape DINO to match L-F1-S3 spatial layout and take first 64 dims
                        D = mr_vol_downsampled.shape[0]
                        dino_mr_reshaped = dino_mr.view(D, 16, 16, -1).reshape(-1, dino_mr.shape[-1])[:, :64]  # Take first 64 dims
                        dino_ct_reshaped = dino_ct.view(D, 16, 16, -1).reshape(-1, dino_ct.shape[-1])[:, :64]
                        
                        # Step 3: Combine L-F1-S3 (32D) + DINO (64D) = 96D total per spatial location
                        f_mr_flat = torch.cat([l_f1_s3_mr.cpu(), dino_mr_reshaped.cpu()], dim=1)  # [N, 96]
                        f_ct_flat = torch.cat([l_f1_s3_ct.cpu(), dino_ct_reshaped.cpu()], dim=1)  # [N, 96]
                        
                        print(f"DEBUG PCA: Hybrid features - MR: {f_mr_flat.shape}, CT: {f_ct_flat.shape}")
                    except Exception as e:
                        print(f"CRITICAL ERROR: Hybrid feature extraction failed during PCA fitting: {e}")
                        raise RuntimeError(f"Hybrid PCA fitting failed: {e}") from e
                else:
                    # Standard DINO features for fallback (256D)
                    f_mr_gpu = dino_encoder.encode_volume(mr_vol_downsampled, use_proj=True, dataset_type=None)
                    f_ct_gpu = dino_encoder.encode_volume(ct_vol_downsampled, use_proj=True, dataset_type=None)
                    f_mr_flat = f_mr_gpu.view(-1, f_mr_gpu.shape[-1]).cpu()
                    f_ct_flat = f_ct_gpu.view(-1, f_ct_gpu.shape[-1]).cpu()
                    print(f"DEBUG PCA: Standard DINO features - MR: {f_mr_flat.shape}, CT: {f_ct_flat.shape}")
            else:
                # Use standard DINO features
                f_mr_gpu = dino_encoder.encode_volume(mr_vol_downsampled, use_proj=True, dataset_type=dataset_type)
                f_ct_gpu = dino_encoder.encode_volume(ct_vol_downsampled, use_proj=True, dataset_type=dataset_type)
                f_mr_flat = f_mr_gpu.view(-1, f_mr_gpu.shape[-1])
                f_ct_flat = f_ct_gpu.view(-1, f_ct_gpu.shape[-1])

            # Create foreground mask for subsampling - OPTIMIZED based on diagnostic results
            if dataset_type and dataset_type.lower() == "abdomen":
                # Based on diagnostic analysis: Use COMBINED approach for best results
                print("Using COMBINED foreground masking strategy for abdomen...")
                
                # Strategy 1: Percentile-based (15th percentile works better than 10th)
                thresh_mr_percentile = torch.quantile(mr_vol_downsampled, 0.15)
                thresh_ct_percentile = torch.quantile(ct_vol_downsampled, 0.15)
                
                # Strategy 2: Otsu thresholding
                try:
                    thresh_mr_otsu = threshold_otsu(mr_vol_downsampled.numpy())
                    thresh_ct_otsu = threshold_otsu(ct_vol_downsampled.numpy())
                except:
                    # Fallback if Otsu fails
                    thresh_mr_otsu = float(thresh_mr_percentile)
                    thresh_ct_otsu = float(thresh_ct_percentile)
                
                # Strategy 3: Conservative statistical approach (less aggressive than before)
                mr_mean, mr_std = mr_vol_downsampled.mean(), mr_vol_downsampled.std()
                ct_mean, ct_std = ct_vol_downsampled.mean(), ct_vol_downsampled.std()
                thresh_mr_statistical = mr_mean - 0.75 * mr_std  # Less aggressive than 0.5
                thresh_ct_statistical = ct_mean - 0.75 * ct_std
                
                # COMBINED approach: Use the median of the three strategies for robustness
                mr_thresholds = [float(thresh_mr_percentile), thresh_mr_otsu, float(thresh_mr_statistical)]
                ct_thresholds = [float(thresh_ct_percentile), thresh_ct_otsu, float(thresh_ct_statistical)]
                
                thresh_mr = sorted(mr_thresholds)[1]  # Take median
                thresh_ct = sorted(ct_thresholds)[1]  # Take median
                
                print(f"  MR thresholds: percentile={thresh_mr_percentile:.3f}, otsu={thresh_mr_otsu:.3f}, statistical={thresh_mr_statistical:.3f}, COMBINED={thresh_mr:.3f}")
                print(f"  CT thresholds: percentile={thresh_ct_percentile:.3f}, otsu={thresh_ct_otsu:.3f}, statistical={thresh_ct_statistical:.3f}, COMBINED={thresh_ct:.3f}")
                
                mr_mask = (mr_vol_downsampled > thresh_mr)
                ct_mask = (ct_vol_downsampled > thresh_ct)
            else:
                # For brain or other datasets: Use conservative percentile-based approach
                thresh_mr = torch.quantile(mr_vol_downsampled, 0.05)  # Very conservative for brain
                thresh_ct = torch.quantile(ct_vol_downsampled, 0.05)
                mr_mask = (mr_vol_downsampled > thresh_mr)
                ct_mask = (ct_vol_downsampled > thresh_ct)

            # Downsample mask to feature resolution
            D, H_feat, W_feat = mr_vol_downsampled.shape[0], 16, 16
            mr_mask_downsampled = F.interpolate(mr_mask.unsqueeze(0).unsqueeze(0).float(), size=(D, H_feat, W_feat), mode='nearest').squeeze().bool()
            ct_mask_downsampled = F.interpolate(ct_mask.unsqueeze(0).unsqueeze(0).float(), size=(D, H_feat, W_feat), mode='nearest').squeeze().bool()
            
            # Apply mask to features
            mr_mask_flat = mr_mask_downsampled.view(-1)
            ct_mask_flat = ct_mask_downsampled.view(-1)

            # Filter to get ONLY foreground features
            f_mr_foreground = f_mr_flat[mr_mask_flat]
            f_ct_foreground = f_ct_flat[ct_mask_flat]

            # Validate and add features if the mask captured meaningful content
            mr_foreground_ratio = mr_mask_flat.float().mean().item()
            ct_foreground_ratio = ct_mask_flat.float().mean().item()
            
            print(f"  Foreground ratios - MR: {mr_foreground_ratio:.3f}, CT: {ct_foreground_ratio:.3f}")
            
            # Check if we have reasonable foreground content (not too much, not too little)
            if (0.1 <= mr_foreground_ratio <= 0.8 and 0.1 <= ct_foreground_ratio <= 0.8 and 
                f_mr_foreground.shape[0] > samples_per_volume and f_ct_foreground.shape[0] > samples_per_volume):
                
                subset_idxs_mr = torch.randperm(f_mr_foreground.shape[0])[:samples_per_volume]
                subset_idxs_ct = torch.randperm(f_ct_foreground.shape[0])[:samples_per_volume]
                all_features.append(f_mr_foreground[subset_idxs_mr].cpu())
                all_features.append(f_ct_foreground[subset_idxs_ct].cpu())
                print(f"  ✅ Added {samples_per_volume} MR and {samples_per_volume} CT foreground features")
            else:
                print(f"  ⚠️  Skipping volume - inadequate foreground (MR: {mr_foreground_ratio:.3f}, CT: {ct_foreground_ratio:.3f})")
                print(f"      Available features - MR: {f_mr_foreground.shape[0]}, CT: {f_ct_foreground.shape[0]}")

    if not all_features:
        raise RuntimeError("No features collected for PCA fitting - all volumes may be empty or feature extraction failed")

    all_features_cat = torch.cat(all_features, dim=0).numpy()
    
    print(f"✅ Fitting PCA on {all_features_cat.shape[0]} feature vectors with {all_features_cat.shape[1]} dimensions...")
    print(f"Expected feature type: {feature_type}, dataset: {dataset_type}")
    
    # Ensure we don't use more PCA components than we have features
    actual_out_dim = min(out_dim, all_features_cat.shape[1], all_features_cat.shape[0])
    if actual_out_dim != out_dim:
        print(f"⚠️  Adjusting PCA dimensions from {out_dim} to {actual_out_dim} due to data constraints")
    
    pca = PCA(n_components=actual_out_dim, svd_solver='randomized')
    pca.fit(all_features_cat)
    print(f"✅ PCA model fitted successfully with {actual_out_dim} components for {feature_type} features.")
    return pca

def train_dino_head(dino_encoder, path_pairs, cache_dir, target_size=(128, 128, 128), head_dataset=None, set_type="train", dataset_type=None):
    """
    Caches features. If is_test_set is True, it saves features for original
    images without augmentation. Otherwise, it creates an augmented dataset.
    """
    proj_head_path = os.path.join(cache_dir, "best_proj_head.pth")
    
    # Train projection head
    if not os.path.exists(proj_head_path) and set_type == "train" and head_dataset is not None:
        print("Starting training for the projection head...")
        dino_encoder.train_proj_head(head_dataset, proj_head_path, epochs=50)

    print(f"Loading pre-trained projection head from {proj_head_path}...")
    dino_encoder.load(proj_head_path)
    dino_encoder.freeze_dino()
    # Also freeze the projection head, as it's already trained
    for param in dino_encoder.proj_head.parameters():
        param.requires_grad = False
    
    # --- PCA Fitting (should only be done on training data) ---
    if set_type == "train":
        # Check if we need to fit separate PCAs for brain dataset
        if dataset_type and dataset_type.lower() == "brain":
            pca_hybrid_path = os.path.join(cache_dir, "pca_hybrid_transformer.pkl")
            pca_standard_path = os.path.join(cache_dir, "pca_transformer.pkl")
            
            # Clean up old incompatible PCA files that might cause dimension mismatches
            old_l_f1_s3_path = os.path.join(cache_dir, "pca_l_f1_s3_transformer.pkl")
            if os.path.exists(old_l_f1_s3_path):
                os.remove(old_l_f1_s3_path)
                print(f"🗑️  Removed old L-F1-S3 PCA file: {old_l_f1_s3_path}")
            
            # Fit PCA for hybrid features (L-F1-S3 + DINO = 96D -> 96D)
            if not os.path.exists(pca_hybrid_path):
                print("🔧 Fitting PCA for HYBRID brain features (L-F1-S3 + DINO)...")
                pca_hybrid_transformer = fit_pca_on_subset(
                    dino_encoder, path_pairs, out_dim=96, target_size=target_size, 
                    dataset_type="brain", feature_type="hybrid"
                )
                joblib.dump(pca_hybrid_transformer, pca_hybrid_path)
                print(f"✅ Saved HYBRID PCA transformer to {pca_hybrid_path}")
            else:
                pca_hybrid_transformer = joblib.load(pca_hybrid_path)
                print(f"✅ Loaded existing HYBRID PCA transformer from {pca_hybrid_path}")
            
            # Also create a fallback standard PCA for when hybrid fails
            if not os.path.exists(pca_standard_path):
                print("🔧 Fitting fallback PCA for standard DINO features...")
                pca_standard_transformer = fit_pca_on_subset(
                    dino_encoder, path_pairs, out_dim=128, target_size=target_size, 
                    dataset_type=None, feature_type="standard"
                )
                joblib.dump(pca_standard_transformer, pca_standard_path)
                print(f"✅ Saved fallback PCA transformer to {pca_standard_path}")
            
            return pca_hybrid_transformer  # Return the hybrid PCA for brain
        else:
            # For non-brain datasets, use standard approach
            pca_path = os.path.join(cache_dir, "pca_transformer.pkl")
            if not os.path.exists(pca_path):
                pca_transformer = fit_pca_on_subset(dino_encoder, path_pairs, out_dim=64, target_size=target_size, dataset_type=dataset_type, feature_type="standard")
                joblib.dump(pca_transformer, pca_path)
            else:
                pca_transformer = joblib.load(pca_path)
            return pca_transformer
    else:
        # For the test set, load the appropriate PCA model fitted on the training data
        if dataset_type and dataset_type.lower() == "brain":
            pca_hybrid_path = os.path.join(cache_dir, "pca_hybrid_transformer.pkl")
            if os.path.exists(pca_hybrid_path):
                print(f"✅ Loading HYBRID PCA transformer for brain test set from {pca_hybrid_path}")
                return joblib.load(pca_hybrid_path)
            else:
                # Fallback to standard PCA if hybrid doesn't exist
                pca_path = os.path.join(cache_dir, "pca_transformer.pkl") 
                print(f"⚠️  Hybrid PCA not found, falling back to standard PCA from {pca_path}")
                return joblib.load(pca_path)
        else:
            pca_path = os.path.join(cache_dir, "pca_transformer.pkl") 
            pca_transformer = joblib.load(pca_path)
            print(f"✅ Loading standard PCA transformer for test set from {pca_path}")
            return pca_transformer

def extract_dino_features(dino_encoder, path_pairs, cache_dir, aug_num, head_dataset=None, set_type="train"):
    """
    Caches features. If is_test_set is True, it saves features for original
    images without augmentation. Otherwise, it creates an augmented dataset.
    """
    os.makedirs(cache_dir, exist_ok=True)
    image_dir = os.path.join(cache_dir, "images")
    feature_dir = os.path.join(cache_dir, "features")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    TARGET_SIZE = (128, 128, 128)
    print(f"--- All volumes will be resampled to {TARGET_SIZE} before feature extraction ---")

    train_dino_head(dino_encoder, path_pairs, cache_dir, target_size=TARGET_SIZE, head_dataset=head_dataset, set_type=set_type)

    mode_dir = os.path.join(cache_dir, set_type)
    image_dir = os.path.join(mode_dir, "images")
    feature_dir = os.path.join(mode_dir, "features")
    os.makedirs(mode_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    # --- Conditional Augmentation ---
    
    if set_type != "train":
        print(f"\nCaching features for {len(path_pairs)} TEST volumes (no augmentation)...")
    else:
        augmentation_transform = get_augmentation_transform()
        print(f"\nCaching features for {len(path_pairs)} TRAIN volumes ({aug_num} augs each)...")
    
    # --- Main Caching Loop ---
    total_saved_count = 0
    for i, (mr_path, ct_path) in enumerate(tqdm(path_pairs, desc="Processing Volumes")):
        original_subject = tio.Subject(
            mr=tio.ScalarImage(mr_path),
            ct=tio.ScalarImage(ct_path)
        )

        for j in range(aug_num):
            with torch.no_grad():
                if set_type != "train":
                    # For the test set, just use the original, un-augmented subject
                    processed_subject = original_subject
                else:
                    # For the training set, create a new augmented version
                    processed_subject = augmentation_transform(original_subject)
                    
                mr_vol_full_res = processed_subject.mr.data.squeeze(0)
                ct_vol_full_res = processed_subject.ct.data.squeeze(0)

                # --- ADD THIS DOWNSAMPLING BLOCK ---
                # Add temporary batch/channel dims to the 3D volume [D, H, W] for interpolation
                mr_vol_unsqueezed = mr_vol_full_res.unsqueeze(0).unsqueeze(0) # Shape -> [1, 1, 192, 192, 192]
                ct_vol_unsqueezed = ct_vol_full_res.unsqueeze(0).unsqueeze(0)

                # Resize the volumes using trilinear interpolation
                mr_vol_downsampled = F.interpolate(mr_vol_unsqueezed, size=TARGET_SIZE, mode='trilinear').squeeze()
                ct_vol_downsampled = F.interpolate(ct_vol_unsqueezed, size=TARGET_SIZE, mode='trilinear').squeeze()
                # The shape is now, for example, [128, 128, 128]
                
                # Foreground masking: Enhanced for intensity-normalized medical images
                if dataset_type and dataset_type.lower() == "abdomen":
                    # For abdomen: Use adaptive thresholding for normalized data
                    mr_mean, mr_std = mr_vol_downsampled.mean(), mr_vol_downsampled.std()
                    ct_mean, ct_std = ct_vol_downsampled.mean(), ct_vol_downsampled.std()
                    
                    # Use statistical thresholding more appropriate for normalized data
                    thresh_mr = mr_mean - 0.5 * mr_std  # Anatomical structures should be above this
                    thresh_ct = ct_mean - 0.5 * ct_std
                    
                    # Also try percentile-based thresholding
                    thresh_mr_percentile = torch.quantile(mr_vol_downsampled, 0.15)
                    thresh_ct_percentile = torch.quantile(ct_vol_downsampled, 0.15)
                    
                    # Use the higher threshold to ensure we get anatomical content
                    thresh_mr = max(float(thresh_mr), float(thresh_mr_percentile))
                    thresh_ct = max(float(thresh_ct), float(thresh_ct_percentile))
                    
                    mr_mask = (mr_vol_downsampled > thresh_mr).float()
                    ct_mask = (ct_vol_downsampled > thresh_ct).float()
                    
                    print(f"  Abdomen foreground thresholds - MR: {thresh_mr:.3f}, CT: {thresh_ct:.3f}")
                elif dataset_type and dataset_type.lower() == "brain":
                    # For brain: Use conservative thresholding to preserve anatomical detail
                    thresh_mr = torch.quantile(mr_vol_downsampled, 0.05)  # Very conservative
                    thresh_ct = torch.quantile(ct_vol_downsampled, 0.05)
                    
                    mr_mask = (mr_vol_downsampled > thresh_mr).float()
                    ct_mask = (ct_vol_downsampled > thresh_ct).float()
                    
                    print(f"  Brain foreground thresholds - MR: {thresh_mr:.3f}, CT: {thresh_ct:.3f}")
                else:
                    # Generic approach using simple intensity thresholding
                    mr_mask = (mr_vol_downsampled > 0.1).float()
                    ct_mask = (ct_vol_downsampled > 0.1).float()
                
                # Apply the masks
                mr_vol_masked = mr_vol_downsampled * mr_mask
                ct_vol_masked = ct_vol_downsampled * ct_mask

                # encode_volume returns [D, N_patches, C], e.g., [128, 256, 256]
                # We permute to [D, H, W] so encode_volume slices along the depth axis.
                f_mr = dino_encoder.encode_volume(mr_vol_masked, use_proj=True)
                f_ct = dino_encoder.encode_volume(ct_vol_masked, use_proj=True)
                
                f_mr_pca_flat = torch.from_numpy(pca_transformer.transform(f_mr.cpu().numpy())).float()
                f_ct_pca_flat = torch.from_numpy(pca_transformer.transform(f_ct.cpu().numpy())).float()
                
                f_mr_pca_flat = F.normalize(f_mr_pca_flat, p=2, dim=-1)
                f_ct_pca_flat = F.normalize(f_ct_pca_flat, p=2, dim=-1)
                
                D = mr_vol_downsampled.shape[0]
                N_PATCHES = 256 # Based on 224x224 input and 14x14 patch size
                H_FEAT = W_FEAT = int(N_PATCHES**0.5) # 16
                C_PCA = f_mr_pca_flat.shape[-1]  # Get actual PCA dimensions (64 or 128)
                
                # Reshape back to 4D feature map [C, D, H_feat, W_feat]
                f_mr_pca = f_mr_pca_flat.view(D, N_PATCHES, C_PCA).view(D, H_FEAT, W_FEAT, C_PCA).permute(3, 0, 1, 2)
                f_ct_pca = f_ct_pca_flat.view(D, N_PATCHES, C_PCA).view(D, H_FEAT, W_FEAT, C_PCA).permute(3, 0, 1, 2)

                # Save with a consistent naming scheme
                save_idx = i * aug_num + j

                # Save the input image for the student
                torch.save(mr_vol_downsampled, os.path.join(image_dir, f"sample_{save_idx}_mr_image.pt"))
                torch.save(ct_vol_downsampled, os.path.join(image_dir, f"sample_{save_idx}_ct_image.pt"))

                # Save the target feature map for the student
                torch.save(f_mr_pca, os.path.join(feature_dir, f"sample_{save_idx}_mr_features.pt"))
                torch.save(f_ct_pca, os.path.join(feature_dir, f"sample_{save_idx}_ct_features.pt"))
                
                total_saved_count += 2 # one for mr, one for ct

    print(f"\nDistillation dataset generation complete. Saved features for {total_saved_count} samples.")
    
def extract_single_dino_features(dino_encoder, mr_vol, ct_vol, pca_transformer, dataset_type=None):
    """
    ENHANCED brain-specific feature extraction with hybrid L-F1-S3 + DINO approach.
    Also includes improved foreground masking for normalized medical images.
    
    For brain: Combines rich L-F1-S3 features (32D) with standard DINO features (64D) for optimal registration.
    Returns features in spatial format [C, D, H, W] for Conv3D compatibility.
    FAILS FAST on any error - no fallbacks to waste compute time.
    """
    with torch.no_grad():
        # CRITICAL FIX: Apply proper intensity normalization for medical images BEFORE feature extraction
        print(f"Raw intensity ranges: MR=[{mr_vol.min():.3f}, {mr_vol.max():.3f}], CT=[{ct_vol.min():.3f}, {ct_vol.max():.3f}]")
        
        def normalize_medical_for_dino(vol):
            """Normalize medical images for optimal DINO feature extraction"""
            # Step 1: Robust intensity normalization (clip outliers)
            p1, p99 = torch.quantile(vol, 0.01), torch.quantile(vol, 0.99)
            vol_clipped = torch.clamp(vol, p1, p99)
            
            # Step 2: Z-score normalization 
            vol_zscore = (vol_clipped - vol_clipped.mean()) / (vol_clipped.std() + 1e-8)
            
            # Step 3: Scale to [0, 1] range for DINO (expects natural image-like inputs)
            vol_min, vol_max = vol_zscore.min(), vol_zscore.max()
            vol_normalized = (vol_zscore - vol_min) / (vol_max - vol_min + 1e-8)
            
            return vol_normalized
        
        # Apply normalization to both volumes
        mr_vol_norm = normalize_medical_for_dino(mr_vol)
        ct_vol_norm = normalize_medical_for_dino(ct_vol)
        
        print(f"Normalized ranges: MR=[{mr_vol_norm.min():.3f}, {mr_vol_norm.max():.3f}], CT=[{ct_vol_norm.min():.3f}, {ct_vol_norm.max():.3f}]")
        
        # Apply dataset-specific foreground masking for better feature quality
        if dataset_type and dataset_type.lower() == "abdomen":
            # Enhanced foreground masking for normalized abdomen images
            print(f"Applying enhanced foreground masking for {dataset_type} dataset")
            
            for vol_name, vol in [("MR", mr_vol), ("CT", ct_vol)]:
                # Multiple thresholding strategies for robust foreground detection
                mean_val, std_val = vol.mean(), vol.std()
                thresh_statistical = mean_val - 0.5 * std_val
                thresh_percentile = torch.quantile(vol, 0.15)  # 15th percentile for abdomen
                
                # Use the higher threshold to ensure anatomical content
                final_thresh = max(float(thresh_statistical), float(thresh_percentile))
                
                # Apply foreground masking
                if vol_name == "MR":
                    mr_vol = mr_vol * (mr_vol > final_thresh).float()
                else:
                    ct_vol = ct_vol * (ct_vol > final_thresh).float()
                
                # Validate foreground content
                foreground_ratio = (vol > final_thresh).float().mean().item()
                print(f"  {vol_name} foreground ratio: {foreground_ratio:.3f} (threshold: {final_thresh:.3f})")
                
                if foreground_ratio < 0.1:
                    print(f"  ⚠️  Warning: Very low {vol_name} foreground ratio - may miss anatomical structures")
                elif foreground_ratio > 0.8:
                    print(f"  ⚠️  Warning: Very high {vol_name} foreground ratio - may include background noise")
        
        elif dataset_type and dataset_type.lower() == "brain":
            print(f"Applying conservative foreground masking for {dataset_type} dataset")
            # Brain-specific conservative masking
            for vol_name, vol in [("MR", mr_vol), ("CT", ct_vol)]:
                thresh_percentile = torch.quantile(vol, 0.05)  # Very conservative for brain
                
                if vol_name == "MR":
                    mr_vol = mr_vol * (mr_vol > thresh_percentile).float()
                else:
                    ct_vol = ct_vol * (ct_vol > thresh_percentile).float()
                
                foreground_ratio = (vol > thresh_percentile).float().mean().item()
                print(f"  {vol_name} foreground ratio: {foreground_ratio:.3f} (threshold: {float(thresh_percentile):.3f})")
        
        # Continue with feature extraction based on dataset type
        if dataset_type and dataset_type.lower() == "brain":
            # HYBRID APPROACH: L-F1-S3 + Standard DINO for richer representation
            print(f"Extracting HYBRID brain features (L-F1-S3 + DINO) for volumes: MR={mr_vol.shape}, CT={ct_vol.shape}")
            
            # Step 1: Extract L-F1-S3 features (32D per spatial location)
            l_f1_s3_mr, l_f1_s3_ct = extract_brain_guided_features(dino_encoder, mr_vol, ct_vol, dataset_type)
            print(f"DEBUG: L-F1-S3 component - MR: {l_f1_s3_mr.shape}, CT: {l_f1_s3_ct.shape}")
            
            # Step 2: Extract standard DINO features for additional representation
            dino_mr = dino_encoder.encode_volume(mr_vol, use_proj=True, dataset_type=None)  # Force standard DINO
            dino_ct = dino_encoder.encode_volume(ct_vol, use_proj=True, dataset_type=None)
            
            # Reshape DINO to match L-F1-S3 spatial layout and take first 64 dims
            D = mr_vol.shape[0]
            dino_mr_reshaped = dino_mr.view(D, 16, 16, -1).reshape(-1, dino_mr.shape[-1])[:, :64]  # Take first 64 dims
            dino_ct_reshaped = dino_ct.view(D, 16, 16, -1).reshape(-1, dino_ct.shape[-1])[:, :64]
            
            print(f"DEBUG: DINO component - MR: {dino_mr_reshaped.shape}, CT: {dino_ct_reshaped.shape}")
            
            # Step 3: Combine L-F1-S3 (32D) + DINO (64D) = 96D total per spatial location
            hybrid_mr = torch.cat([l_f1_s3_mr.cpu(), dino_mr_reshaped.cpu()], dim=1)
            hybrid_ct = torch.cat([l_f1_s3_ct.cpu(), dino_ct_reshaped.cpu()], dim=1)
            
            print(f"DEBUG: Hybrid features - MR: {hybrid_mr.shape}, CT: {hybrid_ct.shape}")
            
            # CRITICAL: Check that PCA transformer expects 96 features for hybrid
            expected_features = pca_transformer.n_features_in_
            actual_features = hybrid_mr.shape[1]
            
            if actual_features != expected_features:
                error_msg = (f"CRITICAL PCA DIMENSION MISMATCH:\n"
                           f"  Hybrid features have {actual_features} dimensions\n" 
                           f"  PCA transformer expects {expected_features} dimensions\n"
                           f"  This indicates the PCA was trained on different feature type\n"
                           f"  FAILING FAST to avoid wasting compute resources")
                print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            print(f"DEBUG: Applying PCA to hybrid features - Input: {hybrid_mr.shape}, PCA expects: {expected_features} features")
            
            # Apply PCA transformation to hybrid features
            f_mr_pca_flat = torch.from_numpy(pca_transformer.transform(hybrid_mr.numpy())).float()
            f_ct_pca_flat = torch.from_numpy(pca_transformer.transform(hybrid_ct.numpy())).float()
            
            # Normalize
            f_mr_pca_flat = F.normalize(f_mr_pca_flat, p=2, dim=-1)
            f_ct_pca_flat = F.normalize(f_ct_pca_flat, p=2, dim=-1)
            
            # Reshape to spatial format: [D*H_feat*W_feat, C_pca] -> [C_pca, D, H_feat, W_feat]
            H_feat = W_feat = 16  # Feature map spatial dimensions
            C_pca = f_mr_pca_flat.shape[-1]  # PCA dimensions after transformation
            
            print(f"Hybrid reshaping: D={D}, H_feat={H_feat}, W_feat={W_feat}, C_pca={C_pca}")
            
            # Reshape: [D*H_feat*W_feat, C_pca] -> [D, H_feat, W_feat, C_pca] -> [C_pca, D, H_feat, W_feat]
            f_mr_pca = f_mr_pca_flat.view(D, H_feat, W_feat, C_pca).permute(3, 0, 1, 2)
            f_ct_pca = f_ct_pca_flat.view(D, H_feat, W_feat, C_pca).permute(3, 0, 1, 2)
            
            print(f"Final HYBRID spatial PCA features - MR: {f_mr_pca.shape}, CT: {f_ct_pca.shape}")
            return f_mr_pca, f_ct_pca
            
        else:
            # For abdomen: Use standard DINO features with improved preprocessing
            f_mr_gpu = dino_encoder.encode_volume(mr_vol, use_proj=True, dataset_type=dataset_type)
            f_ct_gpu = dino_encoder.encode_volume(ct_vol, use_proj=True, dataset_type=dataset_type)

            # Apply PCA transformation
            f_mr_pca_flat = torch.from_numpy(pca_transformer.transform(f_mr_gpu.cpu().numpy())).float()
            f_ct_pca_flat = torch.from_numpy(pca_transformer.transform(f_ct_gpu.cpu().numpy())).float()

            # Normalize
            f_mr_pca_flat = F.normalize(f_mr_pca_flat, p=2, dim=-1)
            f_ct_pca_flat = F.normalize(f_ct_pca_flat, p=2, dim=-1)

            # Standard DINO reshaping: [D * N_patches, C_pca] -> [C_pca, D, H_feat, W_feat]
            D = mr_vol.shape[0]  # Depth dimension from original volume
            N_patches = 256  # 16x16 = 256 patches per slice
            H_feat = W_feat = 16  # Feature map spatial dimensions
            C_pca = f_mr_pca_flat.shape[-1]  # PCA dimensions (64 or 128)
            
            print(f"Standard DINO reshaping: D={D}, N_patches={N_patches}, C_pca={C_pca}")
            
            # Reshape: [D*N_patches, C_pca] -> [D, N_patches, C_pca] -> [D, H_feat, W_feat, C_pca] -> [C_pca, D, H_feat, W_feat]
            f_mr_pca = f_mr_pca_flat.view(D, N_patches, C_pca).view(D, H_feat, W_feat, C_pca).permute(3, 0, 1, 2)
            f_ct_pca = f_ct_pca_flat.view(D, N_patches, C_pca).view(D, H_feat, W_feat, C_pca).permute(3, 0, 1, 2)

            print(f"Final standard DINO spatial PCA features - MR: {f_mr_pca.shape}, CT: {f_ct_pca.shape}")
            return f_mr_pca, f_ct_pca
            
def extract_brain_guided_features(dino_encoder, mr_vol, ct_vol, dataset_type):
    """
    FIXED L-F1-S3 brain-specific feature extraction with consistent dimensions.
    
    L-F1-S3 approach:
    - L: Use DINO as spatial priors/attention (not direct features) 
    - F1: Single fusion strategy combining intensity + DINO guidance
    - S3: Brain-specific anatomical features (32 channels total)
    
    Returns 32-dimensional features per spatial location for 64-96 PCA compression.
    """
    # Move input volumes to the correct device BEFORE any processing
    device = dino_encoder.device if hasattr(dino_encoder, 'device') else next(dino_encoder.parameters()).device
    mr_vol = mr_vol.to(device)
    ct_vol = ct_vol.to(device)
    
    D, H, W = mr_vol.shape
    
    print(f"DEBUG: Starting FIXED L-F1-S3 extraction - MR: {mr_vol.shape}, CT: {ct_vol.shape}, device: {device}")
    
    try:
        # Step 1 (L): Extract DINO spatial attention maps as priors with CONSISTENT dimensions
        # Force consistent input size for ALL slices to ensure same patch grid
        target_input_size = (224, 224)  # Standard DINO input size
        target_patch_grid = (16, 16)    # Expected 16x16 patch grid for ViT-L/14
        
        print(f"DEBUG: Processing {D} slices with consistent {target_input_size} input size")
        
        # Process slices in smaller batches to avoid memory issues but ensure consistency
        batch_size = min(32, D)  # Process up to 32 slices at once
        all_mr_attention_maps = []
        all_ct_attention_maps = []
        
        for start_idx in range(0, D, batch_size):
            end_idx = min(start_idx + batch_size, D)
            batch_slices = end_idx - start_idx
            
            # Prepare batch of slices with EXACTLY the same input size
            mr_batch_slices = []
            ct_batch_slices = []
            
            for i in range(start_idx, end_idx):
                # Extract slice and ensure consistent size
                mr_slice = mr_vol[i].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                ct_slice = ct_vol[i].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                # Force resize to EXACT target size for consistency
                mr_slice_resized = F.interpolate(mr_slice, size=target_input_size, mode='bilinear', align_corners=False)
                ct_slice_resized = F.interpolate(ct_slice, size=target_input_size, mode='bilinear', align_corners=False)
                
                mr_batch_slices.append(mr_slice_resized.squeeze(0))  # Remove batch dim for stacking
                ct_batch_slices.append(ct_slice_resized.squeeze(0))
            
            # Stack into batch: [batch_size, 1, 224, 224]
            mr_batch = torch.stack(mr_batch_slices, dim=0)
            ct_batch = torch.stack(ct_batch_slices, dim=0)
            
            with torch.no_grad():
                # Process batch through DINO
                mr_feats_batch = dino_encoder.extract_dino_features(mr_batch)  # [batch_size, N_patches, 1024]
                ct_feats_batch = dino_encoder.extract_dino_features(ct_batch)  # [batch_size, N_patches, 1024]
                
                # Verify EXACT patch count
                expected_patches = 256  # 16x16 for ViT-L/14 on 224x224 input
                assert mr_feats_batch.shape[1] == expected_patches, f"MR patches: {mr_feats_batch.shape[1]} != {expected_patches}"
                assert ct_feats_batch.shape[1] == expected_patches, f"CT patches: {ct_feats_batch.shape[1]} != {expected_patches}"
                
                # Convert to spatial attention maps with EXACT dimensions
                mr_attention = torch.norm(mr_feats_batch, dim=-1)  # [batch_size, 256]
                ct_attention = torch.norm(ct_feats_batch, dim=-1)  # [batch_size, 256]
                
                # FORCE exact reshape to target grid
                mr_attention_spatial = mr_attention.view(batch_slices, target_patch_grid[0], target_patch_grid[1])  # [batch_size, 16, 16]
                ct_attention_spatial = ct_attention.view(batch_slices, target_patch_grid[0], target_patch_grid[1])  # [batch_size, 16, 16]
                
                # Move to CPU and add to lists
                all_mr_attention_maps.append(mr_attention_spatial.cpu())
                all_ct_attention_maps.append(ct_attention_spatial.cpu())
        
        # Concatenate all batches: [D, 16, 16]
        mr_spatial_priors = torch.cat(all_mr_attention_maps, dim=0)
        ct_spatial_priors = torch.cat(all_ct_attention_maps, dim=0)
        
        # Verify final dimensions
        assert mr_spatial_priors.shape == (D, 16, 16), f"MR priors shape: {mr_spatial_priors.shape} != {(D, 16, 16)}"
        assert ct_spatial_priors.shape == (D, 16, 16), f"CT priors shape: {ct_spatial_priors.shape} != {(D, 16, 16)}"
        
        # Normalize spatial priors to [0,1] range
        mr_spatial_priors = (mr_spatial_priors - mr_spatial_priors.min()) / (mr_spatial_priors.max() - mr_spatial_priors.min() + 1e-8)
        ct_spatial_priors = (ct_spatial_priors - ct_spatial_priors.min()) / (ct_spatial_priors.max() - ct_spatial_priors.min() + 1e-8)
        
        print(f"DEBUG: DINO spatial priors - MR: {mr_spatial_priors.shape}, CT: {ct_spatial_priors.shape}")
        
        # Step 2 (F1 + S3): Create brain-guided features using DINO as priors
        def create_l_f1_s3_features(vol, spatial_priors):
            """Create exactly 32-channel L-F1-S3 features for brain registration with FIXED dimensions"""
            vol = vol.to(device)
            spatial_priors = spatial_priors.to(device)
            
            # FIXED: Use exact target dimensions to prevent shape mismatches
            target_shape = (D, 16, 16)  # Force exact dimensions
            
            # Downsample volume to EXACT feature resolution [D, 16, 16]
            vol_feat_res = F.interpolate(
                vol.unsqueeze(0).unsqueeze(0),
                size=target_shape,
                mode='trilinear',
                align_corners=False
            ).squeeze().to(device)
            
            # CRITICAL: Ensure spatial_priors match exact dimensions
            if spatial_priors.shape != target_shape:
                spatial_priors = F.interpolate(
                    spatial_priors.unsqueeze(0).unsqueeze(0),
                    size=target_shape,
                    mode='trilinear', 
                    align_corners=False
                ).squeeze().to(device)
                print(f"DEBUG: Resized spatial priors to {spatial_priors.shape}")
            
            # F1: Single fusion - DINO-guided intensity
            guided_intensity = vol_feat_res * spatial_priors  # Channel 0: [D, 16, 16]
            
            # FIXED S3: Create all features with GUARANTEED consistent dimensions
            # All operations must preserve [D, 16, 16] shape
            
            def safe_diff(tensor, dim):
                """Compute diff while preserving exact tensor dimensions"""
                if dim == 0:
                    return torch.diff(tensor, dim=dim, prepend=tensor[0:1])
                elif dim == 1:
                    return torch.diff(tensor, dim=dim, prepend=tensor[:, 0:1])
                else:  # dim == 2
                    return torch.diff(tensor, dim=dim, prepend=tensor[:, :, 0:1])
            
            def safe_pool(tensor, kernel_size, stride=1, padding=0):
                """Pooling that preserves exact dimensions by using proper padding"""
                tensor_5d = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, 16, 16]
                pooled = F.avg_pool3d(tensor_5d, kernel_size=kernel_size, stride=stride, padding=padding)
                result = pooled.squeeze()  # Back to [D, 16, 16]
                
                # Force exact target shape if needed
                if result.shape != target_shape:
                    result = F.interpolate(
                        result.unsqueeze(0).unsqueeze(0),
                        size=target_shape,
                        mode='trilinear',
                        align_corners=False
                    ).squeeze()
                return result
            
            # Gradient-based features (6 channels) - all [D, 16, 16]
            grad_d = torch.abs(safe_diff(vol_feat_res, dim=0))          # Channel 1
            grad_h = torch.abs(safe_diff(vol_feat_res, dim=1))          # Channel 2
            grad_w = torch.abs(safe_diff(vol_feat_res, dim=2))          # Channel 3
            gradient_mag = grad_d + grad_h + grad_w                     # Channel 4
            signed_grad_d = safe_diff(vol_feat_res, dim=0)              # Channel 5
            signed_grad_h = safe_diff(vol_feat_res, dim=1)              # Channel 6
            
            # Texture and local statistics (5 channels) - all [D, 16, 16]
            local_mean = safe_pool(vol_feat_res, kernel_size=3, padding=1)    # Channel 7
            local_variance = (vol_feat_res - local_mean) ** 2                 # Channel 8
            local_std = torch.sqrt(torch.clamp(local_variance, min=1e-8))     # Channel 9
            
            # Use simple operations that preserve dimensions
            local_range = torch.abs(vol_feat_res - local_mean)                # Channel 10
            local_contrast = local_std / (local_mean.abs() + 1e-8)           # Channel 11
            
            # Edge and boundary detection (3 channels) - all [D, 16, 16]
            # Simple Laplacian using roll (preserves dimensions)
            laplacian = (torch.roll(vol_feat_res, 1, 0) + torch.roll(vol_feat_res, -1, 0) +
                        torch.roll(vol_feat_res, 1, 1) + torch.roll(vol_feat_res, -1, 1) +
                        torch.roll(vol_feat_res, 1, 2) + torch.roll(vol_feat_res, -1, 2) -
                        6 * vol_feat_res)                                    # Channel 12
            edge_enhanced = guided_intensity * (1 + gradient_mag * 0.5)      # Channel 13
            high_freq = vol_feat_res - local_mean                            # Channel 14: Simplified
            
            # Brain tissue-specific features (6 channels) - all [D, 16, 16]
            csf_enhanced = vol_feat_res * (spatial_priors < 0.3).float()     # Channel 15
            tissue_enhanced = vol_feat_res * (spatial_priors > 0.7).float()  # Channel 16
            boundary_enhanced = vol_feat_res * ((spatial_priors > 0.3) & (spatial_priors < 0.7)).float()  # Channel 17
            vol_sigmoid = torch.sigmoid(vol_feat_res * 5)                    # Channel 18
            vol_tanh = torch.tanh(vol_feat_res * 3)                          # Channel 19
            vol_squared = vol_feat_res ** 2                                  # Channel 20
            
            # Interaction terms (4 channels) - all [D, 16, 16]
            intensity_gradient = vol_feat_res * gradient_mag                  # Channel 21
            prior_gradient = spatial_priors * gradient_mag                   # Channel 22
            intensity_texture = vol_feat_res * local_std                     # Channel 23
            prior_intensity = spatial_priors * vol_feat_res                  # Channel 24
            
            # Multi-scale features (3 channels) - all [D, 16, 16]
            vol_scale1 = safe_pool(vol_feat_res, kernel_size=2, padding=1)   # Channel 25
            spatial_scale1 = safe_pool(spatial_priors, kernel_size=2, padding=1)  # Channel 26
            guided_scale1 = vol_scale1 * spatial_scale1                      # Channel 27
            
            # Additional brain-specific features (4 channels) - all [D, 16, 16]
            vol_sqrt = torch.sqrt(torch.clamp(vol_feat_res.abs(), min=1e-8)) # Channel 28
            spatial_priors_raw = spatial_priors                              # Channel 29
            vol_raw = vol_feat_res                                          # Channel 30
            combined_gradient = signed_grad_d + signed_grad_h                # Channel 31
            
            # Collect all 32 features and verify dimensions
            all_features = [
                guided_intensity,           # 0
                grad_d, grad_h, grad_w, gradient_mag, signed_grad_d, signed_grad_h,  # 1-6
                local_mean, local_variance, local_std, local_range, local_contrast,  # 7-11
                laplacian, edge_enhanced, high_freq,                                  # 12-14
                csf_enhanced, tissue_enhanced, boundary_enhanced, vol_sigmoid, vol_tanh, vol_squared,  # 15-20
                intensity_gradient, prior_gradient, intensity_texture, prior_intensity,               # 21-24
                vol_scale1, spatial_scale1, guided_scale1,                                           # 25-27
                vol_sqrt, spatial_priors_raw, vol_raw, combined_gradient                             # 28-31
            ]
            
            # CRITICAL: Verify ALL features have exact same dimensions before stacking
            for i, feat in enumerate(all_features):
                if feat.shape != target_shape:
                    print(f"ERROR: Feature {i} has shape {feat.shape}, expected {target_shape}")
                    # Force resize to correct shape
                    all_features[i] = F.interpolate(
                        feat.unsqueeze(0).unsqueeze(0),
                        size=target_shape,
                        mode='trilinear',
                        align_corners=False
                    ).squeeze()
                    print(f"FIXED: Feature {i} resized to {all_features[i].shape}")
            
            # Stack exactly 32 channels: [D, 16, 16, 32]
            brain_features = torch.stack(all_features, dim=-1)  # Shape: [D, 16, 16, 32]
            
            print(f"DEBUG: Final L-F1-S3 brain features shape: {brain_features.shape}")
            return brain_features
        
        # Create L-F1-S3 features for both modalities
        mr_brain_features = create_l_f1_s3_features(mr_vol, mr_spatial_priors)
        ct_brain_features = create_l_f1_s3_features(ct_vol, ct_spatial_priors)
        
        print(f"DEBUG: L-F1-S3 brain features - MR: {mr_brain_features.shape}, CT: {ct_brain_features.shape}")
        
        # Flatten for PCA: [D*16*16, 32]
        mr_features_flat = mr_brain_features.reshape(-1, 32)
        ct_features_flat = ct_brain_features.reshape(-1, 32)
        
        print(f"DEBUG: Final flattened L-F1-S3 features - MR: {mr_features_flat.shape}, CT: {ct_features_flat.shape}")
        
        return mr_features_flat, ct_features_flat
        
    except Exception as e:
        print(f"CRITICAL ERROR: L-F1-S3 extraction failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"L-F1-S3 brain-guided feature extraction failed: {e}") from e


def save_feature_slice_as_png(feature_map, filename):
    """
    Visualizes a 3D feature map by taking the middle slice and using PCA 
    to reduce the channels to 3 (RGB) for saving as a PNG.

    Args:
        feature_map (torch.Tensor): A feature map of shape [C, D, H, W].
        filename (str): The path to save the output PNG file.
    """
    # 1. Ensure tensor is on the CPU and in float32 format
    feature_map = feature_map.detach().cpu().to(torch.float32)
    
    # 2. Select the middle slice along the depth (D) axis
    middle_slice_idx = feature_map.shape[1] // 2
    slice_2d = feature_map[:, middle_slice_idx, :, :] # Shape: [C, H, W]
    
    # 3. Use PCA to reduce from C channels to 3 channels
    C, H, W = slice_2d.shape
    
    # Reshape for PCA: [C, H*W] -> [H*W, C]
    slice_reshaped = slice_2d.view(C, -1).permute(1, 0).numpy()
    
    # Fit PCA
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(slice_reshaped) # Shape: [H*W, 3]
    
    # 4. Reshape back to an image format and normalize
    # [H*W, 3] -> [3, H*W] -> [3, H, W]
    img_components = torch.from_numpy(principal_components.T).view(3, H, W)
    
    # Normalize each channel to the [0, 1] range for visualization
    for i in range(3):
        channel = img_components[i]
        min_val, max_val = channel.min(), channel.max()
        if max_val > min_val:
            img_components[i] = (channel - min_val) / (max_val - min_val)
            
    # 5. Save the image
    vutils.save_image(img_components, filename)
    print(f"✅ Saved feature visualization to {filename}")
