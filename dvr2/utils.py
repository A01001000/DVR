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
        
def apply_flow_scaling(flow, from_shape, to_shape, voxel_spacing=None, target_spacing=1.0):
    """Apply both voxel spacing and spatial resolution scaling"""
    scaled_flow = flow.clone()
    
    # 1. Scale for voxel spacing
    if voxel_spacing is not None:
        spacing_factors = torch.tensor([target_spacing / vs for vs in voxel_spacing], device=flow.device, dtype=flow.dtype)
        for i in range(3):
            scaled_flow[:, i] *= spacing_factors[i]
    
    # 2. Scale for spatial resolution
    spatial_factors = torch.tensor([to_shape[i] / from_shape[i] for i in range(3)], device=flow.device)
    for i in range(3):
        scaled_flow[:, i] *= spatial_factors[i]
    
    return scaled_flow

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
