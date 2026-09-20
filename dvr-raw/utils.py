import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from sklearn.decomposition import PCA
import numpy as np
from torch.utils.data import Dataset
import torchio as tio
import os
import numpy as np
import nibabel as nib
from glob import glob
import torchio as tio
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint


def apply_pca_to_flat_features(f_mr, f_ct, out_dim=64):
    """
    Applies PCA to input tensors of shape [N, C]:
        N: number of slices (axial slices from 3D volume) x number of patch tokens per slice (e.g. 196 for 14×14 grid)
        C: DINOv2 embedding dimension (ViT-L/14); 1024
    Returns: tensors of shape [N, out_dim] and the fitted PCA object:
        N: number of axial slices in the volume x number of DINO patch tokens per slice
        out_dim: number of PCA dimensions (e.g., 64)
    """
    f_mr_np = f_mr.detach().cpu().numpy()
    f_ct_np = f_ct.detach().cpu().numpy()
    concat = np.concatenate([f_mr_np, f_ct_np], axis=0)

    pca = PCA(n_components=out_dim)
    transformed = pca.fit_transform(concat) # Use fit_transform here

    f_mr_pca = transformed[:f_mr_np.shape[0]]
    f_ct_pca = transformed[f_mr_np.shape[0]:]

    return torch.from_numpy(f_mr_pca).float().to(f_mr.device), \
           torch.from_numpy(f_ct_pca).float().to(f_ct.device), \
           pca # <-- Return the fitted pca object

def load_nifti(path):
    return torch.tensor(nib.load(path).get_fdata(), dtype=torch.float32)

def prepare_train_dataset(data_dir):
    path_pairs = []
    patients = sorted(os.listdir(data_dir))
    for pid in patients:
        try:
            mr_path = glob(os.path.join(data_dir, pid, "MR", "*.nii.gz"))[0]
            ct_path = glob(os.path.join(data_dir, pid, "CT", "*.nii.gz"))[0]
            path_pairs.append((mr_path, ct_path))
        except IndexError:
            print(f"Warning: Could not find MR/CT pair for patient {pid}. Skipping.")
    return path_pairs

def prepare_test_dataset(data_dir):
    vol_pairs = []
    path_pairs = []
    seg_vol_pairs = []
    seg_path_pairs = []
    patients = sorted(os.listdir(data_dir))
    for pid in patients:
        try:
            mr_path = glob(os.path.join(data_dir, pid, "MR", "*.nii.gz"))[0]
            ct_path = glob(os.path.join(data_dir, pid, "CT", "*.nii.gz"))[0]
            mr_seg_path = glob(os.path.join(data_dir, pid, "MR_seg", "*.nii.gz"))[0]
            ct_seg_path = glob(os.path.join(data_dir, pid, "CT_seg", "*.nii.gz"))[0]
            
            path_pairs.append((mr_path, ct_path))
            seg_path_pairs.append((mr_seg_path, ct_seg_path))

            # mr_vol = load_nifti(mr_path)
            # ct_vol = load_nifti(ct_path)
            # mr_seg = load_nifti(mr_seg_path)
            # ct_seg = load_nifti(ct_seg_path)

            # vol_pairs.append((mr_vol, ct_vol))
            # seg_vol_pairs.append((mr_seg, ct_seg))
        except IndexError:
            print(f"Warning: Could not find MR/CT pair for patient {pid}. Skipping.")

    return path_pairs, seg_path_pairs
    
"""
On-the-fly data augmentation for sparse datasets
"""
def get_augmentation_transform():
    """Creates a pipeline of strong 3D augmentations using TorchIO."""
    return tio.Compose([
        # Spatial Transforms
        tio.RandomAffine(scales=(0.9, 1.2), degrees=15, translation=10, p=0.75),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=12, p=0.75),
        tio.RandomFlip(axes=(0, 1, 2), p=0.25), # Use integer axes for tio
        
        
        # Intensity Transforms
        tio.RandomNoise(p=0.5),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),
        tio.RandomBlur(std=(0, 1.5), p=0.25),
    ])

def extract_3d_patches(img_tensor, patch_size):
    """
    Extracts 3D patches from a 3D image tensor using as_strided.
    This is a highly efficient, zero-copy operation.

    Args:
        img_tensor (torch.Tensor): The input tensor of shape (B, 1, D, H, W).
        patch_size (int): The size of the cubic patch (e.g., 7 for a 7x7x7 patch).

    Returns:
        torch.Tensor: A view of the tensor with patches, shape (B, D, H, W, p, p, p).
    """
    # Ensure patch size is odd
    assert patch_size % 2 == 1, "patch_size must be an odd number."
    
    # Pad the spatial dimensions
    padding = patch_size // 2
    padded_tensor = F.pad(img_tensor, (padding, padding, padding, padding, padding, padding))
    
    # Get the shape and strides of the padded tensor
    B, C, D_pad, H_pad, W_pad = padded_tensor.shape
    sB, sC, sD, sH, sW = padded_tensor.stride()

    # The shape of the output tensor with the patch dimensions
    output_shape = (B, C, D_pad - 2*padding, H_pad - 2*padding, W_pad - 2*padding, patch_size, patch_size, patch_size)
    
    # The strides for the new dimensions
    output_strides = (sB, sC, sD, sH, sW, sD, sH, sW)

    # Create the strided view and remove the channel dimension
    patches = torch.as_strided(padded_tensor, size=output_shape, stride=output_strides).squeeze(1)
    
    return patches

class MINDLoss(nn.Module):
    """
    An implementation of the Modality Independent Neighbourhood Descriptor (MIND)
    for robust multimodal image registration.
    """
    def __init__(self, non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_sigma=3.0):
        super().__init__()
        self.nl_size = non_local_region_size
        self.p_size = patch_size
        self.n_size = neighbor_size
        self.sigma = gaussian_sigma
        
        # Create Gaussian kernel for weighting patch differences
        kernel_tensor = self.create_gaussian_kernel(self.sigma, self.nl_size)
        self.register_buffer('gaussian_kernel', kernel_tensor)

    def create_gaussian_kernel(self, sigma, kernel_size):
        # Generates a 3D Gaussian kernel
        mean = (kernel_size - 1) / 2.0
        x, y, z = np.ogrid[-mean:mean+1, -mean:mean+1, -mean:mean+1]
        g = np.exp(- (x*x + y*y + z*z) / (2 * sigma * sigma))
        g[g < np.finfo(g.dtype).eps * g.max()] = 0
        g_sum = g.sum()
        if g_sum != 0:
            g /= g_sum
        return torch.from_numpy(g).float()

    def pdist(self, x, y):
        # Efficiently computes pairwise squared Euclidean distance
        x_norm = (x**2).sum(1).view(-1, 1)
        y_norm = (y**2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return dist

    def forward(self, img1, img2):
        # Ensure kernel is on the correct device
        self.gaussian_kernel = self.gaussian_kernel.to(img1.device)

        # Compute MIND descriptors for both images
        mind1 = self.compute_mind(img1)
        mind2 = self.compute_mind(img2)
        
        # Calculate L1 distance between descriptors
        return torch.mean(torch.abs(mind1 - mind2))

    def compute_mind(self, img):
        # 1. Extract patches using unfold
        # The input image is (B, 1, D, H, W)
        B, C, D, H, W = img.shape
        patches = extract_3d_patches(img, self.p_size) # (B, D, H, W, p, p, p)
        
        # 2. Get central pixel intensity
        central_pixel = patches[:, :, :, :, self.p_size//2, self.p_size//2, self.p_size//2].unsqueeze(-1)
        
        # 3. Calculate intensity differences
        patches_flat = patches.reshape(B, D, H, W, -1)
        diffs = patches_flat - central_pixel
        
        # 4. Compute variance over the patch
        variance = torch.mean(diffs**2, dim=-1) + 1e-8
        
        # 5. Weight differences by Gaussian kernel and compute MIND descriptor
        # The Gaussian kernel is applied implicitly by the variance calculation in the original paper
        # We apply an exponential function to the negative squared differences, divided by the variance
        mind_features = torch.exp(-diffs**2 / variance.unsqueeze(-1))
        
        # 7. Final descriptor is the mean of the weighted features
        mind_descriptor = mind_features.permute(0, 4, 1, 2, 3)
        return mind_descriptor

def get_warped_grid(flow, size):
    B, C, D, H, W = flow.shape
    Dt,Ht,Wt = size
    # resize flow to this resolution
    flow = F.interpolate(flow, size=(Dt,Ht,Wt), mode='trilinear', align_corners=True)
    dz = torch.linspace(-1,1,Dt,device=flow.device)
    dy = torch.linspace(-1,1,Ht,device=flow.device)
    dx = torch.linspace(-1,1,Wt,device=flow.device)
    grid_z, grid_y, grid_x = torch.meshgrid(dz,dy,dx,indexing='ij')
    grid = torch.stack((grid_x,grid_y,grid_z),dim=3)[None,...].repeat(B,1,1,1,1)

    flow = flow.permute(0,2,3,4,1)
    flow_scaled = torch.zeros_like(flow)
    flow_scaled[...,0] = 2*flow[...,0]/(Wt-1)
    flow_scaled[...,1] = 2*flow[...,1]/(Ht-1)
    flow_scaled[...,2] = 2*flow[...,2]/(Dt-1)
    return grid + flow_scaled

def save_slice_as_png(tensor_5d, filename, slice_dim=2, slice_index=None):
    """
    Saves a 2D slice from a 5D (B, C, D, H, W) tensor as a PNG image.

    Args:
        tensor_5d (torch.Tensor): The 5D tensor, likely on a GPU.
        filename (str): The path to save the PNG file.
        slice_dim (int): The dimension to slice along (2=Depth, 3=Height, 4=Width).
        slice_index (int, optional): The index of the slice. Defaults to the middle slice.
    """
    # Ensure the tensor is on the CPU and remove batch/channel dimensions
    tensor_3d = tensor_5d.detach().squeeze().cpu()

    # If no slice index is provided, choose the middle one
    if slice_index is None:
        slice_index = tensor_3d.shape[slice_dim - 2] // 2

    # Select the 2D slice
    if slice_dim == 2:   # Depth slice
        img_slice = tensor_3d[slice_index, :, :]
    elif slice_dim == 3: # Height slice
        img_slice = tensor_3d[:, slice_index, :]
    elif slice_dim == 4: # Width slice
        img_slice = tensor_3d[:, :, slice_index]
    else:
        raise ValueError("slice_dim must be 2, 3, or 4")

    # Save the image using matplotlib
    plt.imsave(filename, img_slice.numpy(), cmap='gray')
    print(f"✅ Saved debug slice to {filename}")

def inspect_actor_layers(policy, state):
    """
    Performs a final, definitive autopsy on the stable Encoder-Decoder actor.
    """
    print("\n--- FINAL ACTOR LAYER AUTOPSY ---")
    try:
        # Helper for checkpointing
        def run_block(block, x):
            return checkpoint(block, x) if (policy.use_checkpointing and policy.training) else block(x)

        # --- Encoder Pass ---
        x = run_block(policy.planner_conv1, state)
        print(f"L1 (planner_conv1) output norm: {torch.norm(x).item():.2f}")
        x = run_block(policy.planner_conv2, x)
        print(f"L2 (planner_conv2) output norm: {torch.norm(x).item():.2f}")
        x = run_block(policy.planner_conv3, x)
        print(f"L3 (planner_conv3) output norm: {torch.norm(x).item():.2f}")
        x = run_block(policy.planner_conv4, x)
        print(f"L4 (planner_conv4) output norm: {torch.norm(x).item():.2f}")
        x = run_block(policy.planner_conv5, x)
        print(f"L5 (planner_conv5) output norm: {torch.norm(x).item():.2f}")
        bottleneck = run_block(policy.plan_output, x)
        print(f"L6 (plan_output/bottleneck) output norm: {torch.norm(bottleneck).item():.2f}")

        # --- Decoder Pass ---
        x = run_block(policy.actor_upconv1, bottleneck)
        print(f"L7 (actor_upconv1) output norm: {torch.norm(x).item():.2f}")
        x = run_block(policy.actor_upconv2, x)
        print(f"L8 (actor_upconv2) output norm: {torch.norm(x).item():.2f}")
        x = run_block(policy.actor_upconv3, x)
        print(f"L9 (actor_upconv3) output norm: {torch.norm(x).item():.2f}")
        x = run_block(policy.actor_upconv4, x)
        print(f"L10 (actor_upconv4) output norm: {torch.norm(x).item():.2f}")
        x = run_block(policy.actor_upconv5, x)
        print(f"L11 (actor_upconv5) output norm: {torch.norm(x).item():.2f}")

        # --- Final Head ---
        raw_mean = policy.actor_mean(x)
        print(f"L12 FINAL (actor_mean) output norm: {torch.norm(raw_mean).item():.2f}")
        print("----------------------------------\n")
    except Exception as e:
        print(f"--- ERROR DURING AUTOPSY ---: {e}")
        
def apply_flow_scaling(flow_batch, source_shape, target_shape, batch_spacing=None):
    """
    CORRECTED: Scales a batch of flow fields based on the change in resolution,
    handling a list of spacings for the batch.
    
    Args:
        flow_batch: Flow field tensor of shape [B, 3, D, H, W]
        source_shape: Spatial dimensions of source (D, H, W) - should be 3 values
        target_shape: Spatial dimensions of target (D, H, W) - should be 3 values
        batch_spacing: Optional list of spacings for each batch item
    """
    # Ensure we have exactly 3 spatial dimensions
    if len(source_shape) != 3 or len(target_shape) != 3:
        raise ValueError(f"source_shape and target_shape must have exactly 3 dimensions (D, H, W). "
                        f"Got source_shape: {source_shape} (len={len(source_shape)}), "
                        f"target_shape: {target_shape} (len={len(target_shape)})")
    
    # Get the spatial dimensions (D, H, W)
    source_dims = torch.tensor(source_shape, device=flow_batch.device, dtype=torch.float32)
    target_dims = torch.tensor(target_shape, device=flow_batch.device, dtype=torch.float32)

    # Calculate the resizing factor
    resize_factors = (target_dims / source_dims).view(1, 3, 1, 1, 1)

    # --- THE FIX ---
    # The original function had a logic error here. This new logic correctly
    # calculates and applies the scaling for each item in the batch.
    
    # Check if we need to apply spacing correction
    if batch_spacing is not None:
        # Create a scaling tensor for the whole batch
        # Shape will be [B, 3, 1, 1, 1] for broadcasting
        spacing_tensor = torch.tensor(batch_spacing, device=flow_batch.device, dtype=flow_batch.dtype).view(-1, 3, 1, 1, 1)
        
        # Invert the spacing tensor to get the correct scaling factor
        # We scale by (target_spacing / source_spacing), which is what the resize_factors represent
        # if the spacing was isotropic 1.0. Here we adjust for the real spacing.
        # This implementation assumes target spacing is isotropic 1.0 after resampling.
        # If not, you'd pass target_spacing as an argument as well.
        scaling_factors = resize_factors * (1.0 / spacing_tensor)
    else:
        # If no spacing is provided, just use the resize factor
        scaling_factors = resize_factors

    # Apply the scaling to the entire flow batch
    return flow_batch * scaling_factors

# In your training loop, calculate effective spacing after downsampling:
def get_effective_spacing(original_spacing, original_size, current_size):
    """Calculate effective voxel spacing after downsampling"""
    return [orig_sp * (orig_sz / curr_sz) for orig_sp, orig_sz, curr_sz in 
            zip(original_spacing, original_size, current_size)]
    
def vlearn_collate_fn_train(batch):
    """
    Custom collate function to handle image tensors and spacing tuples.
    """
    # Separate the components from the batch
    mr_vols = [item[0] for item in batch]
    ct_vols = [item[1] for item in batch]
    spacings = [item[2] for item in batch] # Gathers spacing tuples into a list

    # Stack the image tensors into a single batch tensor
    mr_batch = torch.stack(mr_vols, 0)
    ct_batch = torch.stack(ct_vols, 0)
    
    # Return the collated batch and the list of spacings
    return mr_batch, ct_batch, spacings
    
def vlearn_collate_fn_val(batch):
    """
    Custom collate function to handle image tensors and spacing tuples.
    """
    # Separate the components from the batch
    mr_vols = [item[0] for item in batch]
    ct_vols = [item[1] for item in batch]
    spacings = [item[2] for item in batch] # Gathers spacing tuples into a list
    mr_x_vols = [item[3] for item in batch]
    ct_x_vols = [item[4] for item in batch]

    # Stack the image tensors into a single batch tensor
    mr_batch = torch.stack(mr_vols, 0)
    ct_batch = torch.stack(ct_vols, 0)
    mr_x_batch = torch.stack(mr_x_vols, 0)
    ct_x_batch = torch.stack(ct_x_vols, 0)

    # Return the collated batch and the list of spacings
    return mr_batch, ct_batch, spacings, mr_x_batch, ct_x_batch

def save_pca_feature(f_mr_pca, f_ct_pca, save_path):
    """
    Visualizes the middle slice of the DINO PCA features to check for
    meaningful anatomical representation.
    """
    # --- FIX: Dynamically find the middle slice ---
    # Get the size of the depth dimension (which is 64 in your case)
    depth = f_mr_pca.shape[1] 
    # Calculate the middle index using integer division (e.g., 64 // 2 = 32)
    middle_slice_idx = depth // 2
    
    print(f"Feature map depth is {depth}, visualizing slice #{middle_slice_idx}.")
    
    # Use the calculated index to select the slice
    mr_slice = f_mr_pca[:, middle_slice_idx, :, :].cpu().numpy()
    ct_slice = f_ct_pca[:, middle_slice_idx, :, :].cpu().numpy()
    
    # Take the first 3 channels and rearrange for plotting [H, W, C]
    mr_rgb = mr_slice[:3, :, :].transpose(1, 2, 0)
    ct_rgb = ct_slice[:3, :, :].transpose(1, 2, 0)
    
    # Normalize to [0, 1] for visualization
    mr_rgb = (mr_rgb - mr_rgb.min()) / (mr_rgb.max() - mr_rgb.min())
    ct_rgb = (ct_rgb - ct_rgb.min()) / (ct_rgb.max() - ct_rgb.min())

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(mr_rgb)
    axes[0].set_title("MR DINO Features")
    axes[0].axis('off') # Hide axes for clarity
    axes[1].imshow(ct_rgb)
    axes[1].set_title("CT DINO Features")
    axes[1].axis('off')
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory
    print(f"Saved feature visualization to {save_path}")

def save_pca_feature(f_mr, f_ct, save_path):
    """
    Save PCA feature visualizations for debugging.
    Handles both 2D [N, C] and 4D [C, D, H, W] tensor formats.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Check tensor dimensions and handle appropriately
    print(f"DEBUG - Feature shapes: MR={f_mr.shape}, CT={f_ct.shape}")
    
    if f_mr.dim() == 2 and f_ct.dim() == 2:
        # 2D features: [N, C] - reshape for visualization
        print("Handling 2D features [N, C] format")
        
        # For 2D features, we'll create a simple summary visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot first 3 PCA components as histograms
        mr_np = f_mr.cpu().numpy()
        ct_np = f_ct.cpu().numpy()
        
        for i in range(min(3, f_mr.shape[1])):
            axes[i].hist(mr_np[:, i], bins=50, alpha=0.7, label='MR', color='blue')
            axes[i].hist(ct_np[:, i], bins=50, alpha=0.7, label='CT', color='red')
            axes[i].set_title(f'PCA Component {i+1}')
            axes[i].legend()
            axes[i].set_xlabel('Feature Value')
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Saved 2D feature histogram visualization to {save_path}")
        
    elif f_mr.dim() == 4 and f_ct.dim() == 4:
        # 4D features: [C, D, H, W] - original logic
        print("Handling 4D features [C, D, H, W] format")
        
        mid_depth = f_mr.shape[1] // 2  # Middle slice along depth (D) dimension
        print(f"Feature map depth is {f_mr.shape[1]}, visualizing axial slice #{mid_depth}.")
        
        # Take axial slice: [C, H, W]
        mr_slice = f_mr[:, mid_depth, :, :]  
        ct_slice = f_ct[:, mid_depth, :, :]
        
        # Use first 3 channels as RGB, or create a summary if more channels
        if mr_slice.shape[0] >= 3:
            mr_rgb = mr_slice[:3].permute(1, 2, 0)  # [H, W, 3]
            ct_rgb = ct_slice[:3].permute(1, 2, 0)  # [H, W, 3]
        else:
            # If fewer than 3 channels, repeat the first channel
            mr_rgb = mr_slice[0:1].repeat(3, 1, 1).permute(1, 2, 0)
            ct_rgb = ct_slice[0:1].repeat(3, 1, 1).permute(1, 2, 0)
        
        # Normalize to [0, 1] for visualization
        mr_rgb = (mr_rgb - mr_rgb.min()) / (mr_rgb.max() - mr_rgb.min() + 1e-8)
        ct_rgb = (ct_rgb - ct_rgb.min()) / (ct_rgb.max() - ct_rgb.min() + 1e-8)
        
        # Create side-by-side visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(mr_rgb.cpu().numpy())
        axes[0].set_title("MR DINO Features (axial)")
        axes[0].axis('off')
        
        axes[1].imshow(ct_rgb.cpu().numpy())
        axes[1].set_title("CT DINO Features (axial)")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Saved 4D feature visualization to {save_path}")
        
    else:
        print(f"ERROR: Unsupported feature dimensions - MR: {f_mr.shape}, CT: {f_ct.shape}")
        print("Expected either 2D [N, C] or 4D [C, D, H, W] format")
        return

def create_registration_state(f_mr_warped, f_ct_fixed, flow_accumulated, flow_scale=5.0):
    """
    Creates a robustly normalized state by scaling each component independently.
    Handles both raw images (3D) and DINO features (4D).
    """
    with torch.no_grad():
        # Check dimensions and handle accordingly
        if f_mr_warped.ndim == 3 and f_ct_fixed.ndim == 3:
            # Raw images case: [D, H, W]
            # Add channel dimension to make them [1, D, H, W] for consistency
            f_mr_warped = f_mr_warped.unsqueeze(0)
            f_ct_fixed = f_ct_fixed.unsqueeze(0)
        elif f_mr_warped.ndim == 4 and f_ct_fixed.ndim == 4:
            # DINO features case: [C, D, H, W] - already correct
            pass
        else:
            raise ValueError(f"Mismatched dimensions: f_mr_warped.ndim={f_mr_warped.ndim}, f_ct_fixed.ndim={f_ct_fixed.ndim}")
        
        # Ensure flow has the same spatial dimensions as the features
        if flow_accumulated.shape != f_mr_warped.shape:
            # Flow should be [3, D, H, W], features should be [C, D, H, W]
            # Make sure spatial dimensions match
            if flow_accumulated.shape[1:] != f_mr_warped.shape[1:]:
                flow_accumulated = F.interpolate(
                    flow_accumulated.unsqueeze(0), 
                    size=f_mr_warped.shape[1:], 
                    mode='trilinear', 
                    align_corners=False
                ).squeeze(0)
        
        # --- ROBUST NORMALIZATION ---
        # Normalize each component to have zero mean and unit variance independently.
        f_mr_norm = (f_mr_warped - f_mr_warped.mean()) / (f_mr_warped.std() + 1e-8)
        f_ct_norm = (f_ct_fixed - f_ct_fixed.mean()) / (f_ct_fixed.std() + 1e-8)
        
        # Flow is already on a reasonable scale, just divide by a constant
        flow_norm = flow_accumulated / flow_scale
        
        # Concatenate the consistently scaled components along channel dimension
        state = torch.cat([f_mr_norm, f_ct_norm, flow_norm], dim=0)
        
        return state
