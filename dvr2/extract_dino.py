import joblib
import torch
import os
from glob import glob
import argparse
from tqdm import tqdm
from sklearn.decomposition import PCA
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

def fit_pca_on_subset(dino_encoder, path_pairs, out_dim=64, samples_per_volume=100, target_size=(128, 192, 192)):
    """
    Fits a PCA model on a random subset of features AFTER they have
    passed through the trained projection head.
    THIS VERSION DOWNSAMPLES THE VOLUMES FIRST.
    """
    print("Collecting a subset of features from the projection head to fit PCA...")
    print(f"--- PCA fitting will use downsampled volumes of size {target_size} ---")
    all_features = []
    
    dino_encoder.proj_head.eval()

    for mr_path, ct_path in tqdm(path_pairs, desc="Collecting PCA Samples"):
        mr_vol_full = torch.tensor(nib.load(mr_path).get_fdata(), dtype=torch.float32)
        ct_vol_full = torch.tensor(nib.load(ct_path).get_fdata(), dtype=torch.float32)

        with torch.no_grad():
            # --- ADD THIS DOWNSAMPLING BLOCK ---
            mr_vol_unsqueezed = mr_vol_full.unsqueeze(0).unsqueeze(0)
            ct_vol_unsqueezed = ct_vol_full.unsqueeze(0).unsqueeze(0)
            
            mr_vol_downsampled = F.interpolate(mr_vol_unsqueezed, size=target_size, mode='trilinear').squeeze()
            ct_vol_downsampled = F.interpolate(ct_vol_unsqueezed, size=target_size, mode='trilinear').squeeze()
            
            # --- USE THE DOWNSAMPLED VOLUMES ---
            f_mr_flat = dino_encoder.encode_volume(mr_vol_downsampled, use_proj=True)
            f_ct_flat = dino_encoder.encode_volume(ct_vol_downsampled, use_proj=True)
            
            f_mr_flat = torch.nn.functional.normalize(f_mr_flat, p=2, dim=-1)
            f_ct_flat = torch.nn.functional.normalize(f_ct_flat, p=2, dim=-1)

        subset_idxs_mr = torch.randperm(f_mr_flat.shape[0])[:samples_per_volume]
        subset_idxs_ct = torch.randperm(f_ct_flat.shape[0])[:samples_per_volume]
        all_features.append(f_mr_flat[subset_idxs_mr].cpu())
        all_features.append(f_ct_flat[subset_idxs_ct].cpu())

    all_features_cat = torch.cat(all_features, dim=0).numpy()
    
    print(f"Fitting PCA on {all_features_cat.shape[0]} feature vectors of dimension {all_features_cat.shape[1]}...")
    pca = PCA(n_components=out_dim, svd_solver='randomized')
    pca.fit(all_features_cat)
    print("PCA model fitted.")
    return pca

def train_dino_head(dino_encoder, path_pairs, cache_dir, target_size=(128, 128, 128), head_dataset=None, set_type="train"):
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
    if set_type == "train" and not os.path.exists(os.path.join(cache_dir, "pca_transformer.pkl")):
        pca_transformer = fit_pca_on_subset(dino_encoder, path_pairs, out_dim=64, target_size=target_size)
        # Save the PCA model to apply it consistently to the test set later
        joblib.dump(pca_transformer, os.path.join(cache_dir, "pca_transformer.pkl"))
    else:
        # For the test set, load the PCA model fitted on the training data
        pca_path = os.path.join(cache_dir, "pca_transformer.pkl") 
        pca_transformer = joblib.load(pca_path)

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
                
                # foreground masking: These thresholds are common starting points but may need tuning for your specific data.
                # The goal is to create a binary mask of the patient's body.
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
                
                # Reshape back to 4D feature map [C, D, H_feat, W_feat]
                f_mr_pca = f_mr_pca_flat.view(D, N_PATCHES, -1).view(D, H_FEAT, W_FEAT, -1).permute(3, 0, 1, 2)
                f_ct_pca = f_ct_pca_flat.view(D, N_PATCHES, -1).view(D, H_FEAT, W_FEAT, -1).permute(3, 0, 1, 2)

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
    
def extract_single_dino_features(dino_encoder, mr_vol, ct_vol, pca_transformer):
    """
    Caches features individually.
    """
    # Use torch.no_grad() to prevent any graph creation within this function
    with torch.no_grad():
        # Masking remains the same
        mr_mask = (mr_vol > 0.1).float()
        ct_mask = (ct_vol > 0.1).float()
        mr_vol_masked = mr_vol * mr_mask
        ct_vol_masked = ct_vol * ct_mask

        # Feature extraction remains the same
        f_mr_gpu = dino_encoder.encode_volume(mr_vol_masked, use_proj=True)
        f_ct_gpu = dino_encoder.encode_volume(ct_vol_masked, use_proj=True)
        
        # --- START MEMORY-SAFE CONVERSION ---
        # Explicitly move to CPU and convert to NumPy
        f_mr_cpu_numpy = f_mr_gpu.cpu().numpy()
        f_ct_cpu_numpy = f_ct_gpu.cpu().numpy()

        # Immediately delete the large GPU tensors now that we have the NumPy versions
        del f_mr_gpu, f_ct_gpu

        # Perform PCA transformation on NumPy arrays
        f_mr_pca_flat_numpy = pca_transformer.transform(f_mr_cpu_numpy)
        f_ct_pca_flat_numpy = pca_transformer.transform(f_ct_cpu_numpy)

        # Immediately delete the intermediate NumPy arrays
        del f_mr_cpu_numpy, f_ct_cpu_numpy

        # Convert back to tensors (they will be on the CPU initially)
        f_mr_pca_flat = torch.from_numpy(f_mr_pca_flat_numpy).float()
        f_ct_pca_flat = torch.from_numpy(f_ct_pca_flat_numpy).float()
        
        # Immediately delete the final NumPy arrays
        del f_mr_pca_flat_numpy, f_ct_pca_flat_numpy
        # --- END MEMORY-SAFE CONVERSION ---
        
        # Normalization and reshaping remain the same
        f_mr_pca_flat = F.normalize(f_mr_pca_flat, p=2, dim=-1)
        f_ct_pca_flat = F.normalize(f_ct_pca_flat, p=2, dim=-1)
        
        D = mr_vol.shape[0]
        N_PATCHES = 256
        H_FEAT = W_FEAT = int(N_PATCHES**0.5)
        
        f_mr_pca = f_mr_pca_flat.view(D, N_PATCHES, -1).view(D, H_FEAT, W_FEAT, -1).permute(3, 0, 1, 2)
        f_ct_pca = f_ct_pca_flat.view(D, N_PATCHES, -1).view(D, H_FEAT, W_FEAT, -1).permute(3, 0, 1, 2)
        
        # Clean up final intermediate tensors before returning
        del f_mr_pca_flat, f_ct_pca_flat

    # The final tensors will be on the CPU; they get moved to the GPU in your main loop.
    return f_mr_pca, f_ct_pca

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