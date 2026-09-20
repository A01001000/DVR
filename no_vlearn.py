import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
import os
import time
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import TensorDataset, DataLoader
import joblib

# Add randomization to break deterministic behavior
torch.manual_seed(np.random.randint(0, 10000))  # Random seed each run
np.random.seed(np.random.randint(0, 10000))
random.seed(np.random.randint(0, 10000))

import os
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
import sys
print("sys.path:", sys.path)

# Note: You will need to install medpy for the HD95 calculation
# pip install medpy
from medpy.metric.binary import hd95

import joblib
from dvr2.extract_dino import extract_single_dino_features

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import collections
import random
import os
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
import argparse
import csv
import time
import gc
import joblib

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
if alloc_conf:
    print(f"✅ PyTorch CUDA Allocator Conf is set to: {alloc_conf}")
else:
    print("❌ WARNING: PYTORCH_CUDA_ALLOC_CONF is NOT set.")

from sklearn.model_selection import train_test_split
from dvr2.dino_encoder import DINOEncoder
from dvr2.test import evaluate_vlearn
from dvr2.utils import prepare_train_dataset, prepare_test_dataset, get_augmentation_transform, vlearn_collate_fn_train, vlearn_collate_fn_val
from dvr2.extract_dino import extract_dino_features, train_dino_head
from dvr2.train_student import train_student_model
from dvr2.dataset import TestDataset, VlearnDataset, HeadDataset
from dvr2.student_extractor import StudentEncoder3D


def gradient_loss(flow):
    """
    Calculates the smoothness of the deformation field by penalizing the L2 norm
    of its spatial gradients.
    """
    dz = flow[:, 2, 1:, :, :] - flow[:, 2, :-1, :, :]
    dy = flow[:, 1, :, 1:, :] - flow[:, 1, :, :-1, :]
    dx = flow[:, 0, :, :, 1:] - flow[:, 0, :, :, :-1]
    loss = (dz**2).mean() + (dy**2).mean() + (dx**2).mean()
    return loss

def normalized_cross_correlation(x, y, eps=1e-8):
    """
    Compute normalized cross correlation between two tensors.
    Returns a value between -1 and 1, where 1 means perfect correlation.
    """
    # Flatten spatial dimensions
    x_flat = x.view(x.size(0), -1)
    y_flat = y.view(y.size(0), -1)
    
    # Center the data
    x_mean = x_flat.mean(dim=1, keepdim=True)
    y_mean = y_flat.mean(dim=1, keepdim=True)
    
    x_centered = x_flat - x_mean
    y_centered = y_flat - y_mean
    
    # Compute correlation
    numerator = (x_centered * y_centered).sum(dim=1)
    denominator = torch.sqrt((x_centered**2).sum(dim=1) * (y_centered**2).sum(dim=1) + eps)
    
    ncc = numerator / denominator
    return ncc.mean()

def anatomical_consistency_loss(warped_features, fixed_features, dataset_type="brain"):
    """
    Brain-specific anatomical consistency loss that encourages proper alignment
    of anatomical structures by looking at feature gradients and local patterns.
    """
    if dataset_type.lower() != "brain":
        return torch.tensor(0.0, device=warped_features.device)
    
    # Compute feature gradients to capture anatomical boundaries
    def compute_feature_gradients(features):
        # Features: [B, C, D, H, W]
        grad_d = torch.diff(features, dim=2)  # Along depth
        grad_h = torch.diff(features, dim=3)  # Along height  
        grad_w = torch.diff(features, dim=4)  # Along width
        return grad_d, grad_h, grad_w
    
    # Get gradients for both feature maps
    warped_grad_d, warped_grad_h, warped_grad_w = compute_feature_gradients(warped_features)
    fixed_grad_d, fixed_grad_h, fixed_grad_w = compute_feature_gradients(fixed_features)
    
    # Align gradient dimensions by taking minimum size
    min_d = min(warped_grad_d.size(2), fixed_grad_d.size(2))
    min_h = min(warped_grad_h.size(3), fixed_grad_h.size(3))
    min_w = min(warped_grad_w.size(4), fixed_grad_w.size(4))
    
    # Compute gradient similarity losses
    grad_loss_d = F.mse_loss(warped_grad_d[:, :, :min_d], fixed_grad_d[:, :, :min_d])
    grad_loss_h = F.mse_loss(warped_grad_h[:, :, :, :min_h], fixed_grad_h[:, :, :, :min_h]) 
    grad_loss_w = F.mse_loss(warped_grad_w[:, :, :, :, :min_w], fixed_grad_w[:, :, :, :, :min_w])
    
    # Add brain-specific structural consistency loss
    try:
        # Use local coherence to ensure nearby features remain similar
        kernel_size = 3
        padding = 1
        
        # Apply 3D average pooling for local smoothing
        warped_smooth = F.avg_pool3d(warped_features, kernel_size=kernel_size, stride=1, padding=padding)
        fixed_smooth = F.avg_pool3d(fixed_features, kernel_size=kernel_size, stride=1, padding=padding)
        
        # Structural consistency: warped and fixed should have similar local structure
        structure_loss = F.mse_loss(warped_smooth, fixed_smooth)
        
        return (grad_loss_d + grad_loss_h + grad_loss_w + structure_loss * 0.1) / 3.1
    except Exception as e:
        # Fallback to just gradient losses if structural loss fails
        print(f"Warning: Structural consistency loss failed: {e}")
        return (grad_loss_d + grad_loss_h + grad_loss_w) / 3.0

def dice_score(pred_seg, true_seg, labels):
    """
    Calculates the Dice score for each label. Assumes pred_seg and true_seg are NumPy arrays.
    """
    scores = {}
    for label in labels:
        pred_mask = (pred_seg == label)
        true_mask = (true_seg == label)
        intersection = np.sum(pred_mask & true_mask)
        denominator = np.sum(pred_mask) + np.sum(true_mask)
        if denominator == 0:
            scores[label] = 1.0 # Both masks are empty
        else:
            scores[label] = 2. * intersection / denominator
    return scores

def jacobian_determinant(disp):
    """
    Calculates the determinant of the Jacobian of the transformation grid.
    A value <= 0 indicates a fold in the deformation.
    disp: (D, H, W, 3) NumPy array representing the displacement field.
    """
    # Calculate gradients of each displacement component
    grad_u = np.gradient(disp[..., 0])
    grad_v = np.gradient(disp[..., 1])
    grad_w = np.gradient(disp[..., 2])

    # Jacobian of the displacement field
    J_disp = np.array([
        [grad_u[0], grad_v[0], grad_w[0]],
        [grad_u[1], grad_v[1], grad_w[1]],
        [grad_u[2], grad_v[2], grad_w[2]]
    ])
    
    # The transformation is T(x) = x + disp(x).
    # The Jacobian of the transformation is J_T = I + J_disp.
    J_xx, J_yx, J_zx = J_disp[0, 0], J_disp[0, 1], J_disp[0, 2]
    J_xy, J_yy, J_zy = J_disp[1, 0], J_disp[1, 1], J_disp[1, 2]
    J_xz, J_yz, J_zz = J_disp[2, 0], J_disp[2, 1], J_disp[2, 2]

    det = (1 + J_xx) * ((1 + J_yy) * (1 + J_zz) - J_zy * J_yz) - \
          J_yx * (J_xy * (1 + J_zz) - J_zy * J_xz) + \
          J_zx * (J_xy * J_yz - (1 + J_yy) * J_xz)

    return det

import nibabel as nib

def get_axial_slice_index(image_shape, affine=None):
    """
    Determine which dimension corresponds to the axial plane.
    If affine is provided, use orientation info, otherwise use smallest dimension.
    """
    if affine is not None:
        try:
            # Get orientation from affine matrix
            orientation = nib.orientations.io_orientation(affine)
            # Find which axis corresponds to the inferior-superior direction (typically axial)
            # Look for axis with orientation code 2 (superior-inferior) or -2 (inferior-superior)
            for i, (axis, direction) in enumerate(orientation):
                if abs(axis) == 2:  # Z-axis (superior-inferior)
                    return int(direction)
        except:
            pass
    
    # Fallback: use smallest dimension (common heuristic)
    return np.argmin(image_shape)

def save_middle_slice_overlay(image, seg, out_path, label_cmap='jet', alpha=0.4):
    """Save overlay exactly like syn_test.py - always use last dimension (Z-axis)"""
    mid = image.shape[2] // 2  # Always use Z dimension (last dimension)
    img_slice = image[:, :, mid]
    seg_slice = seg[:, :, mid]
    
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(img_slice, cmap='gray')
    
    # Only overlay non-background labels (exclude label 0)
    mask = seg_slice > 0
    if np.any(mask):
        colored = np.zeros((*seg_slice.shape, 4))
        colored[mask] = plt.cm.get_cmap(label_cmap)(seg_slice[mask]/np.max(seg_slice[mask]))
        plt.imshow(colored, alpha=alpha)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_overlay_png_masked(image, seg, out_path, label_cmap='jet', alpha=0.5, vmin=None, vmax=None):
    """Save overlay with only organ labels (no background) - exactly like syn_test.py"""
    mid = image.shape[2] // 2  # Always use Z dimension (last dimension) like syn_test.py
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
    """Save deformation field visualization - matches syn_test.py format exactly"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # def_field: (D, H, W, 3) - get upper quarter axial slice to capture more brain
    upper_quarter = int(def_field.shape[2] * 0.75)  # Upper quarter instead of middle
    u = def_field[:, :, upper_quarter, 0]    # Take X-Y plane at upper quarter Z, X component
    v = def_field[:, :, upper_quarter, 1]    # Take X-Y plane at upper quarter Z, Y component
    
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)
    
    # Normalize for HSV (matches syn_test.py exactly)
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
    plt.title('Deformation Field (upper quarter slice)')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_middle_slice_error_map(vol1, vol2, out_path):
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

def precompute_data_in_memory(image_loader, dino_encoder, pca_transformer, device, is_test=False, dataset_type=None, save_dir=None):
    """
    A function to pre-compute and cache data for either training or testing.
    Now fails fast on errors instead of using fallbacks to avoid wasting compute.
    """
    cached_data = []
    desc = "Pre-computing Test Data" if is_test else "Pre-computing Train/Val Features"
    
    print(f"Pre-computing data for {len(image_loader.dataset)} samples...")
    
    # Create PCA features directory if save_dir is provided
    if save_dir:
        pca_features_dir = os.path.join(save_dir, "pca_features")
        os.makedirs(pca_features_dir, exist_ok=True)
    
    for batch_idx, data_batch in enumerate(tqdm(image_loader, desc=desc)):
        if is_test:
            mr_img, ct_img, _, mr_seg, ct_seg = data_batch
        else:
            mr_img, ct_img, _ = data_batch

        # Move to the correct device
        mr_img_device = mr_img.to(device)
        ct_img_device = ct_img.to(device)

        # Get the 3D volume - squeeze the first two dimensions to get [D, H, W] shape
        mr_vol_3d = mr_img_device.squeeze(0).squeeze(0)
        ct_vol_3d = ct_img_device.squeeze(0).squeeze(0)
        
        # Extract features - FAIL FAST on any error
        try:
            f_mr, f_ct = extract_single_dino_features(dino_encoder, mr_vol_3d, ct_vol_3d, pca_transformer, dataset_type=dataset_type)
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Feature extraction failed for batch {batch_idx}: {e}")
            print("💥 FAILING FAST to avoid wasting compute resources")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Feature extraction failed completely for batch {batch_idx}: {e}") from e

        # Validate extracted features
        if f_mr is None or f_ct is None:
            raise RuntimeError(f"Feature extraction returned None for batch {batch_idx} - cannot continue")
            
        if torch.isnan(f_mr).any() or torch.isnan(f_ct).any():
            raise RuntimeError(f"NaN values detected in features for batch {batch_idx} - cannot continue")
        
        # DEBUG: Print feature statistics for first batch
        if batch_idx == 0:
            print(f"DEBUG - Feature shapes: MR={f_mr.shape}, CT={f_ct.shape}")
            print(f"DEBUG - Feature ranges: MR=[{f_mr.min():.4f}, {f_mr.max():.4f}], CT=[{f_ct.min():.4f}, {f_ct.max():.4f}]")
            print(f"DEBUG - Feature means: MR={f_mr.mean():.4f}, CT={f_ct.mean():.4f}")
            print(f"DEBUG - Using dataset_type: {dataset_type}")
            if dataset_type == "brain":
                print(f"DEBUG - Should be using hybrid L-F1-S3+DINO features with proper PCA dimensions")
        
        # Save PCA feature visualization per patient/batch if save_dir is provided
        if save_dir:
            from dvr2.utils import save_pca_feature
            pca_png_path = os.path.join(pca_features_dir, f"patient_{batch_idx}_pca_features.png")
            save_pca_feature(f_mr, f_ct, pca_png_path)
            if batch_idx == 0:
                print(f"✅ Saving PCA feature PNGs per patient to {pca_features_dir}")

        # Store data based on test/train mode
        if is_test:
            # Store everything needed for evaluation
            cached_data.append((
                f_mr.cpu(), f_ct.cpu(),
                mr_img.squeeze(0), ct_img.squeeze(0),
                mr_seg.squeeze(0), ct_seg.squeeze(0)
            ))
        else:
            # Store only features for training - ensure they're in spatial format [C, D, H, W]
            cached_data.append((f_mr.cpu(), f_ct.cpu()))
            
        # Clean up GPU memory
        del mr_img_device, ct_img_device, f_mr, f_ct, mr_vol_3d, ct_vol_3d
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return cached_data

class SpatialTransformer(nn.Module):
    """
    A standard spatial transformer network for warping 3D volumes.
    """
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids).float()
        self.register_buffer('grid', grid.unsqueeze(0))
        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]] # Reorder for grid_sample

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True, padding_mode="border")

class SimpleUNet(nn.Module):
    """
    Proven, simple U-Net that should work well for brain registration.
    Focus on getting >0.88 Dice first, then optimize.
    """
    def __init__(self, in_channels=128, out_channels=3, dataset_type="brain"):
        super(SimpleUNet, self).__init__()
        
        # Simple, proven encoder-decoder
        self.enc1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = self._conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = self._conv_block(128, 256)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(128, 64)
        
        self.up4 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(64, 32)

        # Single flow prediction
        self.flow = nn.Conv3d(32, out_channels, kernel_size=3, padding=1)
        
        self._initialize_weights()

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Very small initialization for flow layer
        nn.init.normal_(self.flow.weight, 0, 1e-5)
        if self.flow.bias is not None:
            nn.init.zeros_(self.flow.bias)

    def forward(self, moving_feat, fixed_feat):
        # Ensure features are in correct 5D format [B, C, D, H, W]
        if moving_feat.dim() == 3:
            # Features are [B, N_spatial, C] -> reshape to [B, C, D, H, W]
            B, N_spatial, C = moving_feat.shape
            # Assuming D=128, H=16, W=16 based on your feature extraction
            D = 128  # This should match your volume depth
            H = W = 16  # Feature map spatial dimensions
            
            if N_spatial == D * H * W:
                moving_feat = moving_feat.transpose(1, 2).view(B, C, D, H, W)
                fixed_feat = fixed_feat.transpose(1, 2).view(B, C, D, H, W)
            else:
                raise ValueError(f"Cannot reshape features: N_spatial={N_spatial}, expected={D*H*W}")
        
        # Concatenate moving and fixed features along channel dimension
        x = torch.cat([moving_feat, fixed_feat], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections
        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e4], dim=1))

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e3], dim=1))

        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d4 = self.up4(d3)
        d4 = self.dec4(torch.cat([d4, e1], dim=1))

        # Flow prediction
        flow = self.flow(d4)
        
        return flow

# --- Training and Testing Functions ---

def validate_registration_quality(model, test_loader, device, epoch, max_samples=2):
    """
    Actually test registration quality on real test data during training.
    This gives us real Dice scores to see if we're improving.
    """
    model.eval()
    
    with torch.no_grad():
        total_dice = 0.0
        sample_count = 0
        
        for i, (moving_feat, fixed_feat, moving_img, fixed_img, moving_seg, fixed_seg) in enumerate(test_loader):
            if i >= max_samples:  # Only test on a few samples to save time
                break
                
            # Move data to device
            moving_feat, fixed_feat = moving_feat.to(device), fixed_feat.to(device)
            moving_img, fixed_img = moving_img.to(device), fixed_img.to(device)
            
            # Get image shape for spatial transformer
            img_shape = moving_img.shape[2:]
            st_image = SpatialTransformer(size=img_shape, mode='bilinear').to(device)
            st_seg = SpatialTransformer(size=img_shape, mode='nearest').to(device)
            
            # Predict flow and upsample to full resolution
            predicted_flow = model(moving_feat, fixed_feat)
            flow_upsampled = F.interpolate(predicted_flow, size=img_shape, mode='trilinear', align_corners=False)
            
            # Warp segmentation
            warped_seg = st_seg(moving_seg.to(device).float(), flow_upsampled).cpu().numpy().squeeze()
            fixed_seg_np = fixed_seg.numpy().squeeze()
            
            # Calculate Dice score
            labels = np.unique(fixed_seg_np)[1:]  # Exclude background
            if len(labels) > 0:
                dices = dice_score(warped_seg, fixed_seg_np, labels)
                avg_dice = np.mean(list(dices.values()))
                total_dice += avg_dice
                sample_count += 1
                
                if i == 0:  # Print detailed info for first sample
                    print(f"    Sample {i+1}: Dice = {avg_dice:.4f}")
                    print(f"    Flow stats: mean={torch.norm(predicted_flow, p=2, dim=1).mean().item():.4f}, max={predicted_flow.abs().max().item():.4f}")
        
        if sample_count > 0:
            avg_dice = total_dice / sample_count
            print(f"  🎯 Validation Dice Score: {avg_dice:.4f} (target: >0.88)")
            
            if avg_dice > 0.85:
                print(f"  🎉 EXCELLENT: Registration is working well!")
            elif avg_dice > 0.75:
                print(f"  ✅ GOOD: Registration is improving")
            elif avg_dice > 0.65:
                print(f"  📊 MODERATE: Some registration progress")
            else:
                print(f"  ⚠️  POOR: Registration may be making things worse")
                
            return avg_dice
    
def debug_registration_quality(model, train_loader, spatial_transformer, device, epoch):
    """
    Check if the model is actually learning to register features properly.
    This helps identify if poor test results are due to training issues.
    """
    model.eval()
    print(f"\n--- DEBUG: Registration Quality Check at Epoch {epoch} ---")
    
    with torch.no_grad():
        # Take first batch for debugging
        moving_features, fixed_features = next(iter(train_loader))
        moving_features, fixed_features = moving_features.to(device), fixed_features.to(device)
        
        # Before registration (identity flow = no movement)
        identity_flow = torch.zeros_like(torch.randn(moving_features.shape[0], 3, *moving_features.shape[2:]))
        identity_flow = identity_flow.to(device)
        identity_warped = spatial_transformer(moving_features, identity_flow)
        identity_loss = F.mse_loss(identity_warped, fixed_features).item()
        
        # After registration (with predicted flow)
        predicted_flow = model(moving_features, fixed_features)
        warped_features = spatial_transformer(moving_features, predicted_flow)
        registration_loss = F.mse_loss(warped_features, fixed_features).item()
        
        # Calculate improvement
        improvement = (identity_loss - registration_loss) / identity_loss * 100
        
        print(f"  Identity (no registration): {identity_loss:.6f}")
        print(f"  After registration: {registration_loss:.6f}")
        print(f"  Improvement: {improvement:.2f}%")
        
        if improvement < 5.0:
            print("  ⚠️  WARNING: Model showing little improvement over identity!")
        elif improvement > 20.0:
            print("  ✅ Good: Model is learning meaningful registration")
        else:
            print("  📊 Moderate: Some learning detected")
            
        # Check flow statistics
        flow_magnitude = torch.norm(predicted_flow, p=2, dim=1).mean().item()
        flow_max = predicted_flow.abs().max().item()
        print(f"  Flow stats: mean={flow_magnitude:.4f}, max={flow_max:.4f}")
        
        if flow_magnitude < 0.1:
            print("  ⚠️  WARNING: Very small flow - model may not be moving features enough")
        elif flow_magnitude > 5.0:
            print("  ⚠️  WARNING: Very large flow - may be causing distortions")
    
    model.train()
    print("--- End Debug Check ---\n")

def train_one_shot(model, train_loader, optimizer, spatial_transformer, device, alpha=1.0, epochs=100, dataset_type="brain", scheduler=None):
    """
    Simple, focused training that should actually work for brain registration.
    No complex progressive training - just solid fundamentals.
    """
    model.train()
    print("--- Starting Focused One-Shot Training ---")
    print(f"Training with alpha={alpha}, lr={optimizer.param_groups[0]['lr']}")
    
    # Add debugging variables
    best_loss = float('inf')
    no_improvement_count = 0
    
    for epoch in range(epochs):
        epoch_loss, epoch_sim_loss, epoch_smooth_loss = 0.0, 0.0, 0.0
        batch_count = 0
        max_flow_magnitude = 0.0
        min_flow_magnitude = float('inf')

        for moving_features, fixed_features in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            moving_features, fixed_features = moving_features.to(device), fixed_features.to(device)

            optimizer.zero_grad()

            predicted_flow = model(moving_features, fixed_features)
            warped_features = spatial_transformer(moving_features, predicted_flow)

            # DEBUG: Check for NaN/Inf in predictions
            if torch.isnan(predicted_flow).any() or torch.isinf(predicted_flow).any():
                print(f"WARNING: NaN/Inf in predicted flow at epoch {epoch+1}")
                continue

            # DEBUG: Track flow statistics
            flow_mag = torch.norm(predicted_flow, p=2, dim=1).mean().item()
            max_flow_magnitude = max(max_flow_magnitude, flow_mag)
            min_flow_magnitude = min(min_flow_magnitude, flow_mag)

            # Use multiple loss components for better registration
            similarity_loss = F.mse_loss(warped_features, fixed_features)
            smoothness_loss = gradient_loss(predicted_flow)
            
            # Add additional losses to encourage meaningful registration
            # 1. Encourage non-zero flow (prevent identity mapping)
            flow_magnitude = torch.norm(predicted_flow, p=2, dim=1).mean()
            magnitude_encouragement = torch.exp(-flow_magnitude * 2.0)  # Higher penalty for very small flows
            
            # 2. Add multi-scale NCC loss for brain (better than single-scale)
            if dataset_type == "brain":
                ncc_loss = 0.0
                for scale in [1.0, 0.5, 0.25]:  # Multi-scale for brain
                    if scale != 1.0:
                        warped_scaled = F.interpolate(warped_features, scale_factor=scale, mode='trilinear')
                        fixed_scaled = F.interpolate(fixed_features, scale_factor=scale, mode='trilinear')
                    else:
                        warped_scaled, fixed_scaled = warped_features, fixed_features
                    
                    scale_ncc = 1.0 - normalized_cross_correlation(warped_scaled, fixed_scaled)
                    ncc_loss += scale_ncc * (scale ** 0.5)  # Weight by scale
                ncc_loss = ncc_loss / 3.0  # Average across scales
            else:
                ncc_loss = 1.0 - normalized_cross_correlation(warped_features, fixed_features)
            
            # 3. Add brain-specific anatomical consistency loss
            anatomical_loss = anatomical_consistency_loss(warped_features, fixed_features, dataset_type=dataset_type)
            
            # DEBUG: Check loss values
            if torch.isnan(similarity_loss) or torch.isnan(smoothness_loss):
                print(f"WARNING: NaN in losses at epoch {epoch+1}")
                print(f"Sim loss: {similarity_loss.item()}, Smooth loss: {smoothness_loss.item()}")
                continue
            
            # Combined loss with multiple components (emphasize anatomical alignment for brain)
            if dataset_type == "brain":
                total_loss = (similarity_loss + ncc_loss * 0.5 + 
                             anatomical_loss * 0.3 +  # Brain-specific anatomical loss
                             alpha * smoothness_loss + 
                             magnitude_encouragement * 0.1)
            else:
                total_loss = (similarity_loss + ncc_loss * 0.5 + 
                             alpha * smoothness_loss + 
                             magnitude_encouragement * 0.1)
            total_loss.backward()
            
            # DEBUG: Check gradients
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            if total_grad_norm > 10.0:
                print(f"WARNING: Large gradient norm {total_grad_norm:.4f} at epoch {epoch+1}")
            
            # Moderate gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_sim_loss += similarity_loss.item()
            epoch_smooth_loss += smoothness_loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        avg_sim = epoch_sim_loss / batch_count
        avg_smooth = epoch_smooth_loss / batch_count
        
        # Enhanced debugging output
        if epoch % 5 == 0 or epoch < 10 or epoch >= epochs - 5:
            print(f"Epoch {epoch+1}: Loss: {avg_loss:.6f}, Sim: {avg_sim:.6f}, Smooth: {avg_smooth:.6f}")
            print(f"  Flow magnitude: [{min_flow_magnitude:.4f}, {max_flow_magnitude:.4f}]")
            print(f"  Gradient norm: {total_grad_norm:.4f}")
            
            # CRITICAL: Check if flow is actually meaningful
            if max_flow_magnitude < 1.0:
                print(f"  ⚠️  WARNING: Very small flow magnitude ({max_flow_magnitude:.4f}) - model may not be registering properly!")
                print(f"      This could explain why Dice scores are not improving.")
            elif max_flow_magnitude > 10.0:
                print(f"  ⚠️  WARNING: Very large flow magnitude ({max_flow_magnitude:.4f}) - may be causing distortions!")
            else:
                print(f"  ✅ Flow magnitude is reasonable for registration")
            
            # Run registration quality check every 10 epochs
            if epoch % 10 == 0:
                debug_registration_quality(model, train_loader, spatial_transformer, device, epoch)
            
            # DEBUG: Check if model is actually learning
            if epoch > 0:
                improvement = (best_loss - avg_loss) / best_loss * 100
                if improvement > 0.1:  # More than 0.1% improvement
                    print(f"  📈 Improved by {improvement:.2f}%")
                    best_loss = avg_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    print(f"  📉 No improvement for {no_improvement_count} epochs")
            else:
                best_loss = avg_loss
        
        # Early stopping if no improvement for many epochs
        if no_improvement_count >= 20:
            print(f"Early stopping at epoch {epoch+1} - no improvement for 20 epochs")
            break
            
        # Early stopping if loss becomes very small
        if avg_loss < 1e-6:
            print(f"Early stopping at epoch {epoch+1} - loss converged")
            break
            
        # DEBUG: Warning if loss is stuck
        if epoch > 10 and avg_loss > 0.01:
            print(f"WARNING: Loss still high ({avg_loss:.6f}) after {epoch+1} epochs - check learning rate or data")
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(avg_loss)

    print("--- Training Finished ---")
    print(f"Final loss: {avg_loss:.6f} (best: {best_loss:.6f})")
    return model

def test_one_shot(model, test_loader, device, results_dir="one_shot_results"):
    """
    A comprehensive testing function that calculates metrics and saves visualizations.
    Works with a loader that provides features, images, and segmentations.
    """
    model.eval()
    os.makedirs(results_dir, exist_ok=True)

    # Metrics accumulators
    all_dices, all_hd95, all_non_pos_jac, all_inference_times = [], [], [], []
    per_label_dices = {}

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- Starting One-Shot Testing ---")
    print(f"Model Parameters: {num_params / 1e6:.2f}M")

    with torch.no_grad():
        for i, (moving_feat, fixed_feat, moving_img, fixed_img, moving_seg, fixed_seg) in enumerate(tqdm(test_loader, desc="Testing")):
            # --- Load Data to Device ---
            moving_feat, fixed_feat = moving_feat.to(device), fixed_feat.to(device)
            moving_img, fixed_img = moving_img.to(device), fixed_img.to(device)

            # Get NumPy arrays for metrics and visualization (they are on CPU)
            moving_seg_np = moving_seg.numpy().squeeze()
            fixed_seg_np = fixed_seg.numpy().squeeze()
            
            # --- DYNAMIC SHAPE AND TRANSFORMER INITIALIZATION ---
            # Get the image shape from the tensor in the current batch
            img_shape = moving_img.shape[2:] # Gets (D, H, W) from [B, C, D, H, W]

            # Initialize SpatialTransformers inside the loop with the correct shape
            st_image = SpatialTransformer(size=img_shape, mode='bilinear').to(device)
            st_seg = SpatialTransformer(size=img_shape, mode='nearest').to(device)

            # --- Inference ---
            start_time = time.time()
            predicted_flow = model(moving_feat, fixed_feat)
            all_inference_times.append(time.time() - start_time)
            
            # --- UPSAMPLE THE FLOW FIELD ---
            # Upsample the flow to match the full image resolution
            flow_upsampled = F.interpolate(predicted_flow, size=img_shape, mode='trilinear', align_corners=False)

            # --- Warping ---
            # Use the upsampled flow to warp the high-resolution images and segmentations
            warped_img = st_image(moving_img, flow_upsampled).cpu().numpy().squeeze()
            warped_seg = st_seg(moving_seg.to(device).float(), flow_upsampled).cpu().numpy().squeeze()
            # --- Metric Calculation ---
            labels = np.unique(fixed_seg_np)[1:] # Exclude background
            if not labels.any(): continue
            
            # Dice Score
            dices = dice_score(warped_seg, fixed_seg_np, labels)
            all_dices.append(np.mean(list(dices.values())))
            for label, score in dices.items():
                if label not in per_label_dices:
                    per_label_dices[label] = []
                per_label_dices[label].append(score)

            # Hausdorff Distance (HD95)
            hd95_score = hd95(warped_seg > 0, fixed_seg_np > 0)
            all_hd95.append(hd95_score)

            # Jacobian Determinant
            flow_np = predicted_flow.cpu().numpy()[0].transpose(1, 2, 3, 0) # D, H, W, 3
            jac_det = jacobian_determinant(flow_np)
            non_pos_jac_percent = np.sum(jac_det <= 0) / jac_det.size * 100
            all_non_pos_jac.append(non_pos_jac_percent)

            # --- Visualization Saving (for every test case) ---
            mid_slice_idx = img_shape[0] // 2
            # Save middle slice overlays using the masked overlay function
            save_overlay_png_masked(fixed_img.cpu().numpy()[0,0], fixed_seg_np, os.path.join(results_dir, f"{i}_fixed_CT_overlay.png"))
            save_overlay_png_masked(moving_img.cpu().numpy()[0,0], moving_seg_np, os.path.join(results_dir, f"{i}_moving_MR_overlay.png"))
            save_overlay_png_masked(warped_img, warped_seg, os.path.join(results_dir, f"{i}_warped_MR_overlay.png"))
            # Save deformation field
            flow_np = predicted_flow.cpu().numpy()[0].transpose(1,2,3,0) # D,H,W,3
            save_middle_slice_deformation_field(flow_np, os.path.join(results_dir, f"{i}_deformation_field_middle.png"))
            # Save error map (on white background, no original image)
            save_middle_slice_error_map(warped_seg, fixed_seg_np, os.path.join(results_dir, f"{i}_error_map_middle.png"))

    # --- Print Final Results ---
    print("\n--- Testing Finished: Final Results ---")
    print(f"Average Dice Score: {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}")
    print(f"Average HD95: {np.mean(all_hd95):.2f} ± {np.std(all_hd95):.2f} mm")
    print(f"Average Non-Positive Jacobian: {np.mean(all_non_pos_jac):.2f}% ± {np.std(all_non_pos_jac):.2f}%")
    print(f"Average Inference Time: {np.mean(all_inference_times):.4f} s")
    print("\nDice Score per Label:")
    for label, scores in per_label_dices.items():
        print(f"  Label {label}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # --- Save Results to CSV ---
    results_csv_path = os.path.join(results_dir, "test_results_summary.csv")
    with open(results_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Metric', 'Mean', 'Std'])
        
        # Write overall metrics
        writer.writerow(['Average_Dice_Score', f"{np.mean(all_dices):.4f}", f"{np.std(all_dices):.4f}"])
        writer.writerow(['Average_HD95_mm', f"{np.mean(all_hd95):.2f}", f"{np.std(all_hd95):.2f}"])
        writer.writerow(['Average_Non_Positive_Jacobian_percent', f"{np.mean(all_non_pos_jac):.2f}", f"{np.std(all_non_pos_jac):.2f}"])
        writer.writerow(['Average_Inference_Time_s', f"{np.mean(all_inference_times):.4f}", f"{np.std(all_inference_times):.4f}"])
        writer.writerow(['Model_Parameters_M', f"{num_params / 1e6:.2f}", "0.00"])
        
        # Write per-label dice scores
        for label, scores in per_label_dices.items():
            writer.writerow([f'Dice_Label_{label}', f"{np.mean(scores):.4f}", f"{np.std(scores):.4f}"])
    
    print(f"\n💾 Results saved to: {results_csv_path}")

def analyze_brain_feature_quality(moving_features, fixed_features, dataset_type="brain"):
    """
    Analyze the quality of brain features to understand registration challenges.
    This helps debug why Dice scores might be poor.
    """
    if dataset_type.lower() != "brain":
        return
        
    print("\n--- 🧠 BRAIN FEATURE ANALYSIS ---")
    print(f"Feature tensor shapes: Moving={moving_features.shape}, Fixed={fixed_features.shape}")
    
    # 1. Feature similarity analysis
    mse_similarity = F.mse_loss(moving_features, fixed_features).item()
    cosine_sim = F.cosine_similarity(
        moving_features.flatten(), 
        fixed_features.flatten(), 
        dim=0
    ).item()
    
    print(f"Feature MSE (lower = more similar): {mse_similarity:.8f}")
    print(f"Feature Cosine Similarity (higher = more similar): {cosine_sim:.4f}")
    
    # 2. Feature diversity analysis
    moving_std = torch.std(moving_features).item()
    fixed_std = torch.std(fixed_features).item()
    print(f"Moving features std dev: {moving_std:.6f}")
    print(f"Fixed features std dev: {fixed_std:.6f}")
    
    # 3. Spatial variation analysis (handle different tensor dimensions)
    if moving_features.dim() >= 5:
        # 5D tensor: [B, C, D, H, W]
        moving_spatial_var = torch.var(moving_features, dim=[2, 3, 4]).mean().item()
        fixed_spatial_var = torch.var(fixed_features, dim=[2, 3, 4]).mean().item()
    elif moving_features.dim() == 4:
        # 4D tensor: [B, C, H, W] 
        moving_spatial_var = torch.var(moving_features, dim=[2, 3]).mean().item()
        fixed_spatial_var = torch.var(fixed_features, dim=[2, 3]).mean().item()
    elif moving_features.dim() == 3:
        # 3D tensor: [B, C, D] or [C, H, W]
        moving_spatial_var = torch.var(moving_features, dim=-1).mean().item()
        fixed_spatial_var = torch.var(fixed_features, dim=-1).mean().item()
    else:
        # 2D tensor or other - compute variance across last dimension
        moving_spatial_var = torch.var(moving_features, dim=-1).mean().item()
        fixed_spatial_var = torch.var(fixed_features, dim=-1).mean().item()
        
    print(f"Moving spatial variation: {moving_spatial_var:.6f}")
    print(f"Fixed spatial variation: {fixed_spatial_var:.6f}")
    
    # 4. Brain-specific feature analysis (only for higher-dim tensors)
    if moving_features.dim() >= 4:
        # Check for anatomical boundary information (gradient magnitude)
        if moving_features.dim() == 5:
            grad_dims = [2, 3, 4]
        elif moving_features.dim() == 4:
            grad_dims = [2, 3]
        else:
            grad_dims = [1]
            
        try:
            moving_grad = torch.gradient(moving_features, dim=grad_dims)
            fixed_grad = torch.gradient(fixed_features, dim=grad_dims)
            
            moving_edge_strength = sum([torch.norm(g).item() for g in moving_grad]) / len(moving_grad)
            fixed_edge_strength = sum([torch.norm(g).item() for g in fixed_grad]) / len(fixed_grad)
            
            print(f"Moving edge information strength: {moving_edge_strength:.6f}")
            print(f"Fixed edge information strength: {fixed_edge_strength:.6f}")
        except Exception as e:
            print(f"Could not compute gradient information: {e}")
            moving_edge_strength = fixed_edge_strength = 0.001  # Default value
    else:
        print("Tensor dimensions too low for gradient analysis")
        moving_edge_strength = fixed_edge_strength = 0.001  # Default value
    
    # 5. Registration difficulty assessment
    if mse_similarity < 1e-6:
        print("🚨 CRITICAL: Features are nearly identical - registration will fail!")
        print("   Suggestion: Using DINO as priors rather than direct features should help")
    elif mse_similarity < 1e-4:
        print("⚠️  WARNING: Features are very similar - challenging for registration")
        print("   Brain-guided feature extraction should provide better discrimination")
    elif cosine_sim > 0.98:
        print("⚠️  WARNING: Very high cosine similarity - features may lack discriminative power")
        print("   Multi-layer DINO fusion should improve feature richness")
    elif moving_spatial_var < 1e-5 or fixed_spatial_var < 1e-5:
        print("⚠️  WARNING: Low spatial variation - features may be too smooth")
        print("   Intensity-based features should add necessary spatial detail")
    elif moving_edge_strength < 1e-3 or fixed_edge_strength < 1e-3:
        print("⚠️  WARNING: Low edge information - may lack anatomical boundaries")
        print("   Brain-guided features should enhance boundary detection")
    else:
        print("✅ Feature characteristics seem reasonable for brain registration")
    
    # 6. Specific recommendations
    if mse_similarity > 1e-3:
        print("💡 RECOMMENDATION: Features have good diversity - focus on registration model")
    else:
        print("💡 RECOMMENDATION: Features may be too similar - enhanced brain extraction should help:")
        print("   ✓ Using DINO as priors rather than direct features")
        print("   ✓ Adding intensity-based features for anatomical detail")
        print("   ✓ Multi-scale feature fusion for different structure sizes")
        print("   ✓ Gradient-based features for anatomical boundaries")
    
    # 7. Success prediction
    if mse_similarity > 1e-4 and moving_spatial_var > 1e-4 and moving_edge_strength > 1e-3:
        print("🎯 PREDICTION: Good chance of achieving >0.85 Dice with these features")
    elif mse_similarity > 1e-5:
        print("📊 PREDICTION: Moderate chance of success - may need training tuning")
    else:
        print("⚠️  PREDICTION: Challenging registration - features may need further enhancement")
    
    print("--- END BRAIN FEATURE ANALYSIS ---\n")

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Device and Argument Parsing ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    parser = argparse.ArgumentParser(description="One-Shot 3D Medical Image Registration")
    parser.add_argument("--train", action="store_true", help="Run the training phase.")
    parser.add_argument("--test", action="store_true", help="Run the evaluation phase.")
    parser.add_argument("--data_name", type=str, default="dataset", help="A general name for the dataset being used.")
    parser.add_argument("--model_name", type=str, default=None, help="Specific name for the model/feature set. Defaults to data_name.")
    parser.add_argument("--train_dir", type=str, default="data/Train", help="Path to the training data folder.")
    parser.add_argument("--train_dir2", type=str, default="data/Train2", help="Path to the second training folder.")
    parser.add_argument("--test_dir", type=str, default="data/Test", help="Path to the test data folder.")
    parser.add_argument("--model_dir", type=str, default="models", help="Root directory to save models and features.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--dataset_type", type=str, default=None, help="Dataset type for DINO layer selection (brain/abdomen).")
    args = parser.parse_args()

    # --- Configuration and Path Setup ---
    ALPHA = 0.01  # Very low regularization for DINO features that lack spatial detail
    BATCH_SIZE = 2   # Reduce batch size for more stable training with problematic features
    LEARNING_RATE = 5e-4  # Lower learning rate for more careful optimization

    model_name = args.model_name if args.model_name else args.data_name
    dino_dir = os.path.join(args.model_dir, "Dino-Features", model_name)
    save_model_dir = os.path.join(args.model_dir, "Ablation", "One-shot", model_name)
    os.makedirs(dino_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)

    # Initialize DINO encoder
    dino_encoder = DINOEncoder(device=device)

    # ==============================================================================
    # === TRAINING BLOCK ===========================================================
    # ==============================================================================
    if args.train:
        print("\n--- 🚀 Starting Training Phase ---")

        # 1. Prepare Datasets and DataLoaders for raw images
        print("Step 1: Preparing image datasets...")
        train_paths = prepare_train_dataset(args.train_dir)
        if args.train_dir2 != "data/Train2":
            train2_paths = prepare_train_dataset(args.train_dir2)
            train_paths += train2_paths
        train_paths, val_paths = train_test_split(train_paths, test_size=0.15, random_state=None)  # Remove fixed seed
        print(f"Data split: {len(train_paths)} training, {len(val_paths)} validation samples.")

        train_dataset_img = VlearnDataset(path_pairs=train_paths, transform=get_augmentation_transform())
        val_dataset_img = VlearnDataset(path_pairs=val_paths, transform=get_augmentation_transform())

        train_image_loader = DataLoader(train_dataset_img, batch_size=1, shuffle=True)  # Add shuffle
        val_image_loader = DataLoader(val_dataset_img, batch_size=1, shuffle=True)     # Add shuffle

        # 2. Load DINO head and PCA transformer
        print("Step 2: Loading DINO head and PCA model...")
        head_dataset = HeadDataset(train_paths, transform=get_augmentation_transform())
        pca_transformer = train_dino_head(dino_encoder, train_paths, dino_dir, target_size=(128, 128, 128), head_dataset=head_dataset, set_type="train", dataset_type=args.dataset_type)
        
        # 3. Pre-compute features for both training and validation sets
        if args.dataset_type and args.dataset_type.lower() == "brain":
            fallback_pca_path = os.path.join(dino_dir, "pca_transformer.pkl")
            if os.path.exists(fallback_pca_path):
                fallback_pca_transformer = joblib.load(fallback_pca_path)
                print(f"Loaded fallback PCA transformer for standard DINO features from {fallback_pca_path}")
            else:
                print("WARNING: No fallback PCA transformer found - creating one now")
                # Train a fallback PCA on standard DINO features  
                from dvr2.extract_dino import fit_pca_on_subset
                fallback_pca_transformer = fit_pca_on_subset(dino_encoder, train_paths, out_dim=128, dataset_type=None)
                joblib.dump(fallback_pca_transformer, fallback_pca_path)
                print(f"Created and saved fallback PCA transformer to {fallback_pca_path}")
        else:
            fallback_pca_transformer = None

        # 3. Pre-compute features for both training and validation sets
        print("Step 3: Pre-computing features from images...")
        train_features_cached = precompute_data_in_memory(train_image_loader, dino_encoder, pca_transformer, device, 
                                                         is_test=False, dataset_type=args.dataset_type, save_dir=save_model_dir)
        val_features_cached = precompute_data_in_memory(val_image_loader, dino_encoder, pca_transformer, device, 
                                                       is_test=False, dataset_type=args.dataset_type)

        # Save PCA feature visualization for comparison (like in vlearn_rob.py)
        if train_features_cached:
            from dvr2.utils import save_pca_feature
            os.makedirs(os.path.join(save_model_dir, "pca_features"), exist_ok=True)
            f_mr_sample = train_features_cached[0][0]  # First training sample MR features
            f_ct_sample = train_features_cached[0][1]  # First training sample CT features
            save_pca_feature(f_mr_sample, f_ct_sample, 
                           os.path.join(save_model_dir, "pca_features", "oneshot_training_features.png"))
            print(f"✅ Saved PCA feature visualization for comparison")

        # 4. Create final DataLoaders for the pre-computed features
        print("Step 4: Creating final feature DataLoaders...")
        train_features_dataset = TensorDataset(
            torch.stack([item[0] for item in train_features_cached]),
            torch.stack([item[1] for item in train_features_cached])
        )
        feature_loader = DataLoader(train_features_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

        # Get PCA dimensions from the actual cached features
        sample_features = train_features_cached[0][0]  # Get first MR features
        pca_dims = sample_features.shape[0]  # Number of PCA dimensions
        input_channels = pca_dims * 2  # Multiply by 2 for moving + fixed features
        print(f"Detected PCA dimensions: {pca_dims}, using {input_channels} input channels")

        print("\n🧹 Clearing memory before training...")
        del train_features_cached, val_features_cached, train_dataset_img, val_dataset_img
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # 5. Initialize Model and Train
        print("Step 5: Initializing model and starting training loop...")
        
        # DEBUG: Check feature data quality before training
        sample_batch = next(iter(feature_loader))
        moving_feat, fixed_feat = sample_batch
        print(f"DEBUG - Training data shapes: Moving={moving_feat.shape}, Fixed={fixed_feat.shape}")
        print(f"DEBUG - Feature ranges: Moving=[{moving_feat.min():.4f}, {moving_feat.max():.4f}], Fixed=[{fixed_feat.min():.4f}, {fixed_feat.max():.4f}]")
        print(f"DEBUG - Feature similarity (should be different): MSE={F.mse_loss(moving_feat, fixed_feat).item():.6f}")
        
        # Check if features are too similar (indicating a problem)
        feature_similarity = F.mse_loss(moving_feat, fixed_feat).item()
        if feature_similarity < 0.001:
            print("⚠️  CRITICAL: Moving and fixed features are nearly identical - this will prevent learning")
            print(f"Current MSE: {feature_similarity:.8f}")
            print("🚨 This suggests there may be a fundamental issue with feature extraction")
        elif feature_similarity < 0.01:
            print(f"⚠️  WARNING: Features are quite similar (MSE={feature_similarity:.6f})")
            print("This is expected for intra-patient MR-CT registration but may make learning challenging")
            print("The model needs to learn subtle differences between MR and CT of the same anatomy")
        elif feature_similarity > 1.0:
            print("✅ Good: Features have substantial differences for registration")
        else:
            print(f"📊 Moderate: Feature differences detected (MSE={feature_similarity:.6f})")
            print("This is reasonable for same-patient MR-CT registration")
        
        # Add brain-specific feature analysis if this is a brain dataset
        if args.dataset_type and args.dataset_type.lower() == "brain":
            analyze_brain_feature_quality(moving_feat, fixed_feat, args.dataset_type)
            
        one_shot_model = SimpleUNet(in_channels=input_channels, dataset_type=args.dataset_type).to(device)
        optimizer = torch.optim.Adam(one_shot_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        sample_feature_shape = feature_loader.dataset.tensors[0].shape[2:] # Get D, H, W of features
        st = SpatialTransformer(size=sample_feature_shape).to(device)

        trained_model = train_one_shot(
            model=one_shot_model,
            train_loader=feature_loader,
            optimizer=optimizer,
            spatial_transformer=st,
            device=device,
            alpha=ALPHA,
            epochs=args.epochs,
            dataset_type=args.dataset_type,
            scheduler=scheduler  # Add scheduler
        )

        # 6. Save the trained model
        model_save_path = os.path.join(save_model_dir, "one_shot_model.pth")
        print(f"\n💾 Saving trained model to {model_save_path}")
        torch.save(trained_model.state_dict(), model_save_path)

    # ==============================================================================
    # === TESTING BLOCK ============================================================
    # ==============================================================================
    if args.test:
        print("\n--- 🔬 Starting Testing Phase ---")

        # 1. Initialize model and load trained weights
        print("Step 1: Initializing model and loading weights...")
        
        # Load PCA transformer first to determine dimensions
        proj_head_path = os.path.join(dino_dir, "best_proj_head.pth")
        if args.dataset_type and args.dataset_type.lower() == "brain":
            # For brain, prioritize loading the hybrid PCA first
            pca_hybrid_path = os.path.join(dino_dir, "pca_hybrid_transformer.pkl")
            if os.path.exists(pca_hybrid_path):
                pca_transformer = joblib.load(pca_hybrid_path)
                print(f"✅ Loaded HYBRID PCA transformer for brain testing: {pca_hybrid_path}")
                print(f"Hybrid PCA: {pca_transformer.n_features_in_}D input -> {pca_transformer.n_components_}D output")
            else:
                # Fallback to standard PCA
                pca_path = os.path.join(dino_dir, "pca_transformer.pkl")
                pca_transformer = joblib.load(pca_path)
                print(f"⚠️  Hybrid PCA not found, loaded standard PCA: {pca_path}")
        else:
            pca_path = os.path.join(dino_dir, "pca_transformer.pkl")
            pca_transformer = joblib.load(pca_path)
            print(f"✅ Loaded standard PCA transformer: {pca_path}")
        
        dino_encoder.load(proj_head_path)
        dino_encoder.freeze_dino()
        
        # Determine input channels from PCA transformer
        pca_dims = pca_transformer.n_components_
        input_channels = pca_dims * 2  # Multiply by 2 for moving + fixed features
        print(f"Detected PCA dimensions: {pca_dims}, using {input_channels} input channels")
        
        # CRITICAL: For brain datasets, verify we're using the correct hybrid PCA dimensions
        if args.dataset_type and args.dataset_type.lower() == "brain":
            # Check if this is a hybrid PCA (expecting 96D features -> 96D PCA output)
            expected_hybrid_input = 96   # L-F1-S3 (32) + DINO (64) = 96 input features
            expected_hybrid_output = 96  # PCA output dimension to match training (96*2 = 192 model input channels)
            
            if pca_transformer.n_features_in_ == expected_hybrid_input:
                print(f"✅ Detected hybrid brain PCA: {expected_hybrid_input}D input -> {pca_dims}D output")
                if pca_dims == expected_hybrid_output:
                    print(f"✅ PCA output dimensions correct for trained model: {pca_dims}D -> {input_channels} input channels")
                else:
                    print(f"⚠️  PCA output dimension mismatch: {pca_dims}D (expected {expected_hybrid_output}D for trained model)")
                    print(f"⚠️  This will likely cause model loading errors!")
            else:
                print(f"⚠️  PCA input features: {pca_transformer.n_features_in_}D (expected {expected_hybrid_input}D for hybrid)")
        
        print(f"🏗️  Initializing model with {input_channels} input channels...")
        model = SimpleUNet(in_channels=input_channels, dataset_type=args.dataset_type).to(device)
        print(f"✅ Model initialized successfully")
        model_save_path = os.path.join(save_model_dir, "one_shot_model.pth")

        if not os.path.exists(model_save_path):
            raise FileNotFoundError(f"❌ Model file not found at {model_save_path}. Please train first with the --train flag.")
        
        print(f"📥 Loading model from {model_save_path}")
        try:
            checkpoint = torch.load(model_save_path, map_location=device)
            model.load_state_dict(checkpoint)
            print(f"✅ Model loaded successfully!")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"❌ CRITICAL: Model architecture mismatch!")
                print(f"Error: {e}")
                print(f"This suggests the saved model was trained with different input dimensions.")
                print(f"Current model expects {input_channels} input channels.")
                print(f"💡 SOLUTION: Either retrain the model or check PCA dimensions match training.")
                raise RuntimeError(f"Model dimension mismatch - retrain required: {e}")
            else:
                raise e

        # 2. Load DINO head and PCA transformer (needed to process test images)
        print("Step 2: Loading DINO head and PCA model...")
        # Already loaded above
        
        # Also load fallback PCA for brain datasets
        if args.dataset_type and args.dataset_type.lower() == "brain":
            fallback_pca_path = os.path.join(dino_dir, "pca_transformer.pkl")
            if os.path.exists(fallback_pca_path):
                fallback_pca_transformer = joblib.load(fallback_pca_path)
                print(f"Loaded fallback PCA transformer for testing: {fallback_pca_path}")
            else:
                fallback_pca_transformer = None
                print("WARNING: No fallback PCA transformer found for testing!")
        else:
            fallback_pca_transformer = None

        # 3. Prepare test data loader for raw images and segmentations
        print("Step 3: Preparing raw test image dataset...")
        test_paths, label_paths = prepare_test_dataset(args.test_dir)
        test_dataset_raw = TestDataset(vol_pairs=test_paths, label_pairs=label_paths)
        test_loader_raw = DataLoader(test_dataset_raw, batch_size=1, shuffle=False)

        # 4. Pre-compute all necessary test data (features, images, segmentations)
        print("Step 4: Pre-computing full data for test set...")
        final_test_data = precompute_data_in_memory(test_loader_raw, dino_encoder, pca_transformer, device, 
                                                   is_test=True, dataset_type=args.dataset_type, 
                                                   save_dir=os.path.join(save_model_dir, f"results_{args.data_name}"))

        # 5. Create the final DataLoader for evaluation
        print("Step 5: Creating final test DataLoader...")
        test_loader_final = DataLoader(final_test_data, batch_size=1, shuffle=False)

        # 6. Run evaluation
        print("Step 6: Running evaluation...")
        test_one_shot(
            model=model,
            test_loader=test_loader_final,
            device=device,
            results_dir=os.path.join(save_model_dir, f"results_{args.data_name}")
        )
    
    # ==============================================================================
    # === ABLATION TESTING BLOCK ===================================================
    # ==============================================================================
    def test_dino_ablations():
        """
        Test different DINO configurations to see if any work better for brain registration
        """
        
        # Test 1: Higher Resolution DINO Features
        # Use larger patch overlap and higher feature resolution
        
        # Test 2: Multi-Scale DINO  
        # Extract features at multiple scales and combine
        
        # Test 3: DINO + Intensity Hybrid
        # Use DINO as attention/weighting for intensity features
        
        # Test 4: Layer-specific DINO
        # Use earlier DINO layers that preserve more spatial information
        
        pass