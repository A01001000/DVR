import torch
import torch.nn.functional as F
import numpy as np
import math
from turtle import done
import torch.nn as nn
from scipy.ndimage import gaussian_filter

class HierarchicalRewardSystem:
    def __init__(self, device, coarse_weight_start=1.0, coarse_weight_end=0.1, 
                 annealing_epochs=25):
        self.device = device
        self.coarse_weight_start = coarse_weight_start
        self.coarse_weight_end = coarse_weight_end
        self.annealing_epochs = annealing_epochs
        
    def compute_cls_alignment_reward(self, f_mr_3d, f_ct_3d):
        """
               dy = torch.mean((flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]) ** 2).item()    Compute coarse alignment reward using CLS-like global features
        """
        with torch.no_grad():
            # Global average pooling on the ORIGINAL UNWARPED features
            mr_global = F.adaptive_avg_pool3d(f_mr_3d.unsqueeze(0), (1, 1, 1)).squeeze()
            ct_global = F.adaptive_avg_pool3d(f_ct_3d.unsqueeze(0), (1, 1, 1)).squeeze()
            
            # Cosine similarity for global alignment
            global_similarity = F.cosine_similarity(mr_global, ct_global, dim=0)
            
            return global_similarity.item()
    
    def compute_fine_alignment_reward(self, f_mr_3d, f_ct_3d, warped_mr_features):
        """
        Compute fine-grained alignment reward using patch-level features
        This captures local anatomical details after coarse alignment
        """
        with torch.no_grad():
            # Multi-scale patch alignment - this is what makes it "fine"
            similarities = []
            
            # Original patch resolution - captures fine details
            sim_full = F.cosine_similarity(
                warped_mr_features.flatten(), 
                f_ct_3d.flatten(), 
                dim=0
            )
            similarities.append(sim_full)
            
            # Slightly pooled patches - captures medium-scale structures
            # This is still "fine" compared to global CLS features
            mr_med = F.avg_pool3d(warped_mr_features.unsqueeze(0), 2, 2).squeeze(0)
            ct_med = F.avg_pool3d(f_ct_3d.unsqueeze(0), 2, 2).squeeze(0)
            sim_med = F.cosine_similarity(mr_med.flatten(), ct_med.flatten(), dim=0)
            similarities.append(sim_med)
            
            # Weighted combination emphasizing finest details
            weights = [0.7, 0.3]  # More weight on finest scale
            fine_reward = sum(w * s.item() for w, s in zip(weights, similarities))
            
            return fine_reward
    
    def get_combined_reward(self, f_mr_3d, f_ct_3d, warped_mr_features, flow_acc, 
                           epoch, total_epochs, smooth_val, mag_val, smooth_weight, mag_weight, sim_weight):
        """
        Combine coarse and fine rewards with curriculum annealing
        Early training: Focus on coarse global alignment (like DINO-Reg rigid step)
        Later training: Focus on fine local deformations
        """
        
        # Call coarse reward with the original unwarped f_mr_3d
        coarse_reward = self.compute_cls_alignment_reward(f_mr_3d, f_ct_3d)
        
        # The fine reward correctly uses the warped features
        fine_reward = self.compute_fine_alignment_reward(f_mr_3d, f_ct_3d, warped_mr_features)
        
        # Annealing: Start with coarse (global alignment), shift to fine (local details)
        warmup_epochs = 10 # Hold coarse_weight at 1.0 for this many epochs

        if epoch < warmup_epochs:
            coarse_weight = 1.0
        elif epoch < self.annealing_epochs:
            # Adjust progress to be from 0 to 1 within the annealing window
            progress = (epoch - warmup_epochs) / (self.annealing_epochs - warmup_epochs)
            cosine_val = 0.5 * (1 + np.cos(np.pi * progress))
            coarse_weight = self.coarse_weight_end + (self.coarse_weight_start - self.coarse_weight_end) * cosine_val
        else:
            coarse_weight = self.coarse_weight_end

        fine_weight = 1.0 - coarse_weight
        
        # Combine rewards
        alignment_reward = coarse_weight * coarse_reward + fine_weight * fine_reward
        
        # Total reward with regularization
        total_reward = 100 * (sim_weight * alignment_reward) - (smooth_weight * smooth_val) - (mag_weight * mag_val)
        
        return total_reward, coarse_reward, fine_reward, coarse_weight

class RewardNormalizer:
    """Calculates a running mean and standard deviation for rewards."""
    def __init__(self, num_inputs, clip=10.0):
        self.clip = clip
        self.n = 0
        self.mean = np.zeros(1)
        self.mean_diff = np.zeros(1)
        self.var = np.zeros(1)

    def update(self, reward):
        self.n += 1
        delta = reward - self.mean
        self.mean += delta / self.n
        self.mean_diff += delta * (reward - self.mean)
        self.var = self.mean_diff / self.n if self.n > 1 else np.square(self.mean)
    
    def normalize(self, rewards_tensor):
        # Detach, move to CPU, and convert to numpy for calculation
        rewards_np = rewards_tensor.detach().cpu().numpy()
        
        # Get the running standard deviation, avoid division by zero
        std = np.sqrt(self.var) + 1e-8
        
        # Normalize, clip to prevent extreme values, and convert back to tensor
        normalized_rewards = np.clip((rewards_np - self.mean) / std, -self.clip, self.clip)
        return torch.tensor(normalized_rewards, dtype=torch.float32, device=rewards_tensor.device)
    
def compute_smoothness_penalty(flow):
    """
    Computes L1 smoothness (total variation) of displacement field.
    flow: (B,3,D,H,W)
    """
    dx = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).mean()
    dy = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]).mean()
    dz = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]).mean()
    
    return dx + dy + dz

def compute_hierarchical_reward(fixed_feats, warped_feats, epoch, max_epochs,
                                smoothness_penalty, magnitude_penalty,
                                smooth_weight=0.5, mag_weight=2.0):
    """
    Computes a hierarchical reward that anneals from coarse (global) to fine (local) similarity.
    """
    # --- Fine-grained Reward (Local Similarity) ---
    # Your original multi_scale_reward is a good measure of fine-grained similarity.
    fine_similarity = multi_scale_reward(fixed_feats, warped_feats)

    # --- Coarse-grained Reward (Global Similarity) ---
    # We create a coarse representation by average pooling the feature maps.
    # This forces the agent to match the overall, low-frequency structure first.
    coarse_fixed = [F.adaptive_avg_pool3d(f, (1, 1, 1)) for f in fixed_feats]
    coarse_warped = [F.adaptive_avg_pool3d(f, (1, 1, 1)) for f in warped_feats]
    coarse_similarity = multi_scale_reward(coarse_fixed, coarse_warped)

    # --- Annealing Schedule ---
    # Start by focusing on the coarse reward, then gradually shift focus to the fine reward.
    # We use a cosine schedule for a smooth transition.
    progress = epoch / max_epochs
    coarse_weight = 0.5 * (1 + math.cos(math.pi * progress)) # Starts at 1.0, ends at 0.0
    fine_weight = 1 - coarse_weight                         # Starts at 0.0, ends at 1.0
    
    # --- Final Combined Reward ---
    similarity_reward = (coarse_weight * coarse_similarity) + (fine_weight * fine_similarity)
    
    total_reward = similarity_reward - (smooth_weight * smoothness_penalty) - (mag_weight * magnitude_penalty)
    
    return total_reward, similarity_reward, coarse_similarity, fine_similarity

def local_cosine(fixed, moving, win=9):
    """Local cosine similarity over patches."""
    unfold = nn.Unfold(kernel_size=win, stride=1, padding=win//2)
    # flatten patches
    f_p = unfold(fixed).transpose(1, 2)  # (B, N_patches, C*win^2)
    m_p = unfold(moving).transpose(1, 2)

    f_p = F.normalize(f_p, dim=-1)
    m_p = F.normalize(m_p, dim=-1)

    sim = (f_p * m_p).sum(-1)  # cosine similarity per patch
    return sim.mean()

def multi_scale_reward(fixed_feats, warped_feats):
    sims = []
    for f_fix, f_warp in zip(fixed_feats, warped_feats):
        f_fix = F.normalize(f_fix,dim=1)
        f_warp = F.normalize(f_warp,dim=1)
        sims.append((f_fix*f_warp).sum(1).mean())
    return torch.stack(sims).mean()

def compute_image_gradients(img):
    """
    Compute gradients along x, y, z for 3D images.
    img: (B, C, D, H, W)
    Returns: (B, C, 3, D, H, W)  [dx, dy, dz]
    """
    dz = img[:, :, 2:, :, :] - img[:, :, :-2, :, :]
    dy = img[:, :, :, 2:, :] - img[:, :, :, :-2, :]
    dx = img[:, :, :, :, 2:] - img[:, :, :, :, :-2]

    # pad to keep same size
    dz = F.pad(dz, (0,0,0,0,1,1))
    dy = F.pad(dy, (0,0,1,1,0,0))
    dx = F.pad(dx, (1,1,0,0,0,0))

    grads = torch.stack([dx, dy, dz], dim=2)
    return grads

def ngf_loss(fixed, moving_warped, eps=1e-5):
    """
    NGF loss: encourages gradient directions to align
    fixed: (B,1,D,H,W) fixed image
    moving_warped: (B,1,D,H,W) warped moving image
    """
    grad_f = compute_image_gradients(fixed)
    grad_m = compute_image_gradients(moving_warped)

    # normalize gradients
    grad_f = grad_f / (torch.norm(grad_f, dim=2, keepdim=True) + eps)
    grad_m = grad_m / (torch.norm(grad_m, dim=2, keepdim=True) + eps)

    # 1 - cosine similarity between gradients
    sim = (grad_f * grad_m).sum(2)  # (B,C,D,H,W)
    loss = 1 - sim.mean()
    return loss

# Alternative: Feature-Image Consistency Check (Optional Addition)
class FeatureImageConsistencyReward:
    def __init__(self, device, consistency_weight=0.1):
        self.device = device
        self.consistency_weight = consistency_weight
        
    def compute_consistency_reward(self, mr_img, ct_img, warped_mr_img, 
                                 f_mr_features, f_ct_features, warped_mr_features):
        """
        Lightweight consistency check between image-level and feature-level similarities
        Uses downsampled images to reduce computational cost
        """
        with torch.no_grad():
            # Downsample images for efficiency (e.g., to 64^3)
            target_size = (64, 64, 64)
            mr_small = F.interpolate(mr_img, size=target_size, mode='trilinear')
            ct_small = F.interpolate(ct_img, size=target_size, mode='trilinear')  
            warped_mr_small = F.interpolate(warped_mr_img, size=target_size, mode='trilinear')
            
            # Compute normalized cross-correlation on downsampled images
            def ncc_small(img1, img2):
                img1_flat = img1.flatten()
                img2_flat = img2.flatten()
                img1_norm = (img1_flat - img1_flat.mean()) / (img1_flat.std() + 1e-8)
                img2_norm = (img2_flat - img2_flat.mean()) / (img2_flat.std() + 1e-8)
                return torch.mean(img1_norm * img2_norm)
            
            image_similarity = ncc_small(warped_mr_small, ct_small)
            
            # Compute feature similarity  
            feature_similarity = F.cosine_similarity(
                warped_mr_features.flatten(), 
                f_ct_features.flatten(), 
                dim=0
            )
            
            # Consistency reward: penalize when image and feature similarities disagree
            consistency = 1.0 - abs(image_similarity.item() - feature_similarity.item())
            
            return self.consistency_weight * consistency

def normalize_advantages(advantages, eps=1e-8):
    """
    Normalize advantages for more stable training (V-Learn paper technique)
    
    Args:
        advantages: Tensor of advantages [batch_size, ...]
        eps: Small epsilon for numerical stability
    
    Returns:
        Normalized advantages with zero mean and unit variance
    """
    if advantages.numel() == 1:
        # Single advantage, return as-is
        return advantages
    
    # Flatten advantages for statistics calculation
    flat_advantages = advantages.view(-1)
    
    # Calculate mean and std
    mean = flat_advantages.mean()
    std = flat_advantages.std() + eps
    
    # Normalize
    normalized = (advantages - mean) / std
    
    # Additional safety check
    if not torch.isfinite(normalized).all():
        print("WARNING: NaN/Inf in advantage normalization. Returning a detached clone of original advantages to prevent leaks.")
        # Return a new tensor with no computation history
        return advantages.clone().detach()
    
    return normalized

class DualChannelRewardSystem:
    def __init__(self, device):
        self.device = device
        # Use simple running stats for normalization, not EMA, for better stability
        self.reward_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}

    def compute_cls_alignment_reward(self, f_mr_3d, f_ct_3d):
        """
        Compute coarse alignment reward using global features. Unchanged.
        """
        with torch.no_grad():
            mr_global = F.adaptive_avg_pool3d(f_mr_3d.unsqueeze(0), (1, 1, 1)).squeeze()
            ct_global = F.adaptive_avg_pool3d(f_ct_3d.unsqueeze(0), (1, 1, 1)).squeeze()
            return F.cosine_similarity(mr_global, ct_global, dim=0).item()

    def mind_descriptor(self, img, radius=1, dilation=1):
        """
        NEW: A robust, differentiable MIND-SSD reward. This is the key fix.
        Lower MIND-SSD (better alignment) should result in a higher reward.
        """
        # This function expects a 5D tensor B, C, D, H, W
        if img.dim() != 5:
            raise ValueError("Input to mind_descriptor must be a 5D tensor")

        # Pad the input tensor
        padded = F.pad(img, (radius, radius, radius, radius, radius, radius), mode='replicate')

        out = []
        # Loop through all shifts in the 3D neighborhood
        for dz in range(-radius, radius + 1, dilation):
            for dy in range(-radius, radius + 1, dilation):
                for dx in range(-radius, radius + 1, dilation):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue

                    # --- THE CORRECTED SLICING LOGIC ---
                    # This uses negative indexing to be more robust
                    start_z, end_z = radius + dz, -(radius - dz) if (radius - dz) != 0 else None
                    start_y, end_y = radius + dy, -(radius - dy) if (radius - dy) != 0 else None
                    start_x, end_x = radius + dx, -(radius - dx) if (radius - dx) != 0 else None
                    
                    shifted = padded[:, :, start_z:end_z, start_y:end_y, start_x:end_x]
                    out.append(shifted)
        
        # Stack the shifted tensors to create the descriptor channels
        out = torch.stack(out, dim=1)  # B, N, C, D, H, W
        
        # Compute the variance-based descriptor
        dist = torch.mean((img.unsqueeze(1) - out) ** 2, dim=1) # Variance across shifts
        dist_var = torch.mean((dist - torch.mean(dist, dim=1, keepdim=True))**2, dim=1) # Variance of variances
        
        # Clamp the variance to prevent division by zero for stable gradients
        dist_var = dist_var.clamp(min=1e-5)
        
        # Final MIND descriptor
        mind = torch.exp(-dist / dist_var)
        return mind
        
    def compute_mind_ssd_reward(self, warped_features, fixed_features):
        """
        NEW: A robust, differentiable MIND-SSD reward.
        This version maps the loss to a reward in the range [0, 1].
        """

        with torch.no_grad():
            mind_warped = self.mind_descriptor(warped_features.unsqueeze(0))
            mind_fixed = self.mind_descriptor(fixed_features.unsqueeze(0))

            # Sum of Squared Differences between the robust MIND features
            mind_ssd_loss = ((mind_warped - mind_fixed) ** 2).mean()

            # --- THE FIX: Map the loss to a reward ---
            # A lower loss (better alignment) will result in a reward closer to 1.0
            # A higher loss (worse alignment) will result in a reward closer to 0.0
            # The 'temperature' parameter (e.g., 0.1) controls the sharpness of the reward.
            temperature = 0.1 
            reward = torch.exp(-mind_ssd_loss / temperature)
            
            return reward.item()

    def get_combined_reward(self, f_mr_3d, f_ct_3d, warped_mr_features, flow_acc,
                           epoch, total_epochs, smooth_val, mag_val, smooth_weight, mag_weight, sim_weight):
        
        coarse_reward = self.compute_cls_alignment_reward(f_mr_3d, f_ct_3d)
        
        # Use the new, stable MIND-SSD reward for fine-tuning
        fine_reward = self.compute_mind_ssd_reward(warped_mr_features, f_ct_3d)
        
        # Use fixed weights. The agent is rewarded for both global and local alignment.
        coarse_weight = 0.3
        fine_weight = 0.7 # Emphasize the more informative MIND-SSD reward

        alignment_reward = coarse_weight * coarse_reward + fine_weight * fine_reward
        
        # Scale rewards to positive range and balance components better
        # Scale alignment reward to [0, 100] range for stability
        scaled_alignment = 100 * max(0, alignment_reward)
        
        # Much gentler penalties - let the agent learn alignment first
        penalty_scale = 0.1  # Very light penalties during learning
        total_reward = scaled_alignment - penalty_scale * (smooth_weight * smooth_val + mag_weight * mag_val)
        
        # Ensure reward stays positive to encourage exploration
        total_reward = max(total_reward, 0.1)
        
        return total_reward, coarse_reward, fine_reward, coarse_weight
    
class SimplifiedSSDRewardSystem:
    def __init__(self, device):
        self.device = device

    def get_reward(self, warped_features, fixed_features, flow_field, smoothness_weight, magnitude_weight):
        """
        A simple, robust reward system based on feature similarity (SSD) and regularization.
        """
        with torch.no_grad():
            # 1. Similarity Reward (lower SSD is better, so we negate it for reward)
            similarity_loss = ((warped_features - fixed_features)**2).mean()
            similarity_reward = -similarity_loss.item()

            # 2. Smoothness Penalty
            dz = flow_field[:, :, 1:, :, :] - flow_field[:, :, :-1, :, :]
            dy = flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]
            dx = flow_field[:, :, :, :, 1:] - flow_field[:, :, :, :, :-1]
            smoothness_penalty = (dz**2).mean() + (dy**2).mean() + (dx**2).mean()
            
            # 3. Magnitude Penalty
            magnitude_penalty = (flow_field**2).mean()

            # Combine the components with a high weight on similarity
            total_reward = (1000 * similarity_reward) - \
                           (smoothness_weight * smoothness_penalty.item()) - \
                           (magnitude_weight * magnitude_penalty.item())

            return total_reward
        
class MindSSDRewardSystem:
    def __init__(self, device):
        self.device = device

    def mind_descriptor(self, img, radius=1, dilation=1):
        """
        Calculates the MIND descriptor for a given feature map.
        This version is corrected to prevent crashes and instability.
        """
        # Expects a 5D tensor B, C, D, H, W
        B, C, D, H, W = img.shape
        
        # Pad the input tensor
        padded = F.pad(img, (radius, radius, radius, radius, radius, radius), mode='replicate')

        out = []
        # Loop through all shifts in the 3D neighborhood
        for dz in range(-radius, radius + 1, dilation):
            for dy in range(-radius, radius + 1, dilation):
                for dx in range(-radius, radius + 1, dilation):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    # Correct, robust slicing
                    shifted = padded[:, :, radius + dz : radius + dz + D, 
                                     radius + dy : radius + dy + H, 
                                     radius + dx : radius + dx + W]
                    out.append(shifted)
        
        out = torch.stack(out, dim=1)  # B, N, C, D, H, W
        
        dist = torch.mean((img.unsqueeze(1) - out) ** 2, dim=1)
        dist_var = torch.mean((dist - torch.mean(dist, dim=1, keepdim=True))**2, dim=1)
        
        # Clamp the variance to prevent division by zero for stable gradients
        dist_var = dist_var.clamp(min=1e-5)
        
        mind = torch.exp(-dist / dist_var)
        return mind

    # In your MindSSDRewardSystem class
    def get_reward(self, warped_features, fixed_features, flow_field, 
                smoothness_weight, magnitude_weight, similarity_weight):
        """
        A simple, robust, cost-based reward system.
        """
        with torch.no_grad():
            mind_warped = self.mind_descriptor(warped_features.unsqueeze(0))
            mind_fixed = self.mind_descriptor(fixed_features.unsqueeze(0))
            
            # 1. Calculate the three separate "costs"
            similarity_cost = ((mind_warped - mind_fixed)**2).mean().item()
            
            dz = flow_field[:, :, 1:, :, :] - flow_field[:, :, :-1, :, :]
            dy = flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]
            dx = flow_field[:, :, :, :, 1:] - flow_field[:, :, :, :, :-1]
            smoothness_cost = (dz.abs()).mean().item() + (dy.abs()).mean().item() + (dx.abs()).mean().item()
            
            magnitude_cost = (flow_field.abs()).mean().item()
            
            # 2. Calculate the total weighted cost
            total_cost = (similarity_weight * similarity_cost) + \
                        (smoothness_weight * smoothness_cost) + \
                        (magnitude_weight * magnitude_cost)
                        
            # 3. The reward is the negative of the total cost. This is a stable signal.
            # The agent's reward is *only* the image similarity.
            # We want to MAXIMIZE similarity. ncc_similarity is [B] (higher is better).
            # We scale it by the similarity_weight.
            total_reward = ncc_similarity * similarity_weight

            # We still return the TENSOR costs for the decoder loss and logging
            return total_reward, similarity_cost, smoothness_cost, magnitude_cost
        
class StableNCCRewardSystem:
    """
    A much simpler, more stable reward system using Normalized Cross-Correlation.
    This provides clearer gradients for deformable registration learning.
    """
    def __init__(self, device):
        self.device = device
        
    def normalized_cross_correlation(self, img1, img2):
        """Compute normalized cross-correlation between two images."""
        # Flatten the images
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        
        # Normalize to zero mean, unit variance
        img1_norm = (img1_flat - img1_flat.mean()) / (img1_flat.std() + 1e-8)
        img2_norm = (img2_flat - img2_flat.mean()) / (img2_flat.std() + 1e-8)
        
        # Compute correlation
        ncc = torch.mean(img1_norm * img2_norm)
        return ncc.item()
    
    def get_reward(self, warped_features, fixed_features, flow_field, 
                   smoothness_weight, magnitude_weight, similarity_weight):
        """Simple, stable reward based on NCC similarity."""
        with torch.no_grad():
            # Compute NCC similarity (range [-1, 1], higher is better)
            ncc_similarity = self.normalized_cross_correlation(warped_features, fixed_features)
            
            # Convert to positive reward (range [0, 2])
            similarity_reward = (ncc_similarity + 1.0) * 50.0  # Scale to [0, 100]
            
            # Simple regularization penalties
            flow_magnitude = torch.norm(flow_field, p=2, dim=1).mean().item()
            
            # Gradient-based smoothness penalty (more meaningful than finite differences)
            dz = flow_field[:, :, 1:, :, :] - flow_field[:, :, :-1, :, :]
            dy = flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]
            dx = flow_field[:, :, :, :, 1:] - flow_field[:, :, :, :, :-1]
            smoothness_cost = (dx.abs().mean() + dy.abs().mean() + dz.abs().mean()).item()
            
            # Magnitude penalty
            magnitude_cost = flow_magnitude
            
            # Adaptive penalty scaling
            if flow_magnitude < 0.5:  # Too little movement - reduce penalties
                penalty_scale = 0.1
            elif flow_magnitude > 15.0:  # Too much movement - increase penalties
                penalty_scale = 3.0
            else:  # Normal range - standard penalties
                penalty_scale = 1.0
            
            smoothness_penalty = smoothness_weight * smoothness_cost * penalty_scale
            magnitude_penalty = magnitude_weight * magnitude_cost * penalty_scale
            
            # Final reward
            total_reward = similarity_reward - smoothness_penalty - magnitude_penalty
            
            return total_reward, similarity_reward, smoothness_cost, magnitude_cost

class StableRegistrationReward:
    """
    Simplified, stable reward system focused specifically on deformable registration.
    Uses normalized cross-correlation which provides clearer learning signals than MIND.
    """
    def __init__(self, device):
        self.device = device
        
    def normalized_cross_correlation(self, img1, img2):
        """Compute normalized cross-correlation - excellent for registration."""
        # Flatten and normalize
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        
        # Zero mean, unit variance normalization
        img1_norm = (img1_flat - img1_flat.mean()) / (img1_flat.std() + 1e-8)
        img2_norm = (img2_flat - img2_flat.mean()) / (img2_flat.std() + 1e-8)
        
        # Correlation coefficient
        ncc = torch.mean(img1_norm * img2_norm)
        return ncc
    
    def get_reward(self, warped_features, fixed_features, flow_field, 
                   smoothness_weight, magnitude_weight, similarity_weight):
        """Simple, stable reward focused on registration quality with enhanced scaling."""
        with torch.no_grad():
            # Primary reward: NCC similarity (range [-1, 1], higher is better)
            ncc_similarity = self.normalized_cross_correlation(warped_features, fixed_features)
            
            # Enhanced similarity reward with better scaling
            # Transform [-1,1] to [0,200] but with exponential scaling for better discrimination
            if ncc_similarity > 0:
                # Exponential scaling for positive correlations to reward good alignments more
                similarity_reward = 100.0 * (1.0 + ncc_similarity) * (1.0 + ncc_similarity * 0.5)
            else:
                # Linear penalty for negative correlations
                similarity_reward = 100.0 * (1.0 + ncc_similarity)
            
            # Simple L2 magnitude penalty
            magnitude_penalty = torch.mean(flow_field ** 2).item()
            
            # Simple smoothness penalty using total variation
            dx = torch.mean((flow_field[:, :, :, :, 1:] - flow_field[:, :, :, :, :-1]) ** 2).item()
            dy = torch.mean((flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]) ** 2).item()
            dz = torch.mean((flow_field[:, :, 1:, :, :] - flow_field[:, :, :-1, :, :]) ** 2).item()
            smoothness_penalty = dx + dy + dz
            
            # Adaptive penalty scaling based on flow magnitude for stability
            flow_magnitude = torch.mean(torch.norm(flow_field, p=2, dim=1)).item()
            penalty_scale = 1.0
            if flow_magnitude < 0.5:  # Too little movement
                penalty_scale = 0.05  # Very light penalties to encourage movement
            elif flow_magnitude > 12.0:  # Too much movement
                penalty_scale = 2.0   # Stronger penalties
            elif flow_magnitude > 8.0:
                penalty_scale = 1.5   # Moderate penalties
                
            # Final reward calculation with clearer scaling
            total_reward = similarity_reward.item() - \
                          (smoothness_weight * smoothness_penalty * penalty_scale) - \
                          (magnitude_weight * magnitude_penalty * penalty_scale)
            
            return total_reward, similarity_reward.item(), smoothness_penalty, magnitude_penalty

import torch
import torch.nn.functional as F
import numpy as np
import math
from turtle import done
import torch.nn as nn
from scipy.ndimage import gaussian_filter

class HierarchicalRewardSystem:
    def __init__(self, device, coarse_weight_start=1.0, coarse_weight_end=0.1, 
                 annealing_epochs=25):
        self.device = device
        self.coarse_weight_start = coarse_weight_start
        self.coarse_weight_end = coarse_weight_end
        self.annealing_epochs = annealing_epochs
        
    def compute_cls_alignment_reward(self, f_mr_3d, f_ct_3d):
        """
               dy = torch.mean((flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]) ** 2).item()    Compute coarse alignment reward using CLS-like global features
        """
        with torch.no_grad():
            # Global average pooling on the ORIGINAL UNWARPED features
            mr_global = F.adaptive_avg_pool3d(f_mr_3d.unsqueeze(0), (1, 1, 1)).squeeze()
            ct_global = F.adaptive_avg_pool3d(f_ct_3d.unsqueeze(0), (1, 1, 1)).squeeze()
            
            # Cosine similarity for global alignment
            global_similarity = F.cosine_similarity(mr_global, ct_global, dim=0)
            
            return global_similarity.item()
    
    def compute_fine_alignment_reward(self, f_mr_3d, f_ct_3d, warped_mr_features):
        """
        Compute fine-grained alignment reward using patch-level features
        This captures local anatomical details after coarse alignment
        """
        with torch.no_grad():
            # Multi-scale patch alignment - this is what makes it "fine"
            similarities = []
            
            # Original patch resolution - captures fine details
            sim_full = F.cosine_similarity(
                warped_mr_features.flatten(), 
                f_ct_3d.flatten(), 
                dim=0
            )
            similarities.append(sim_full)
            
            # Slightly pooled patches - captures medium-scale structures
            # This is still "fine" compared to global CLS features
            mr_med = F.avg_pool3d(warped_mr_features.unsqueeze(0), 2, 2).squeeze(0)
            ct_med = F.avg_pool3d(f_ct_3d.unsqueeze(0), 2, 2).squeeze(0)
            sim_med = F.cosine_similarity(mr_med.flatten(), ct_med.flatten(), dim=0)
            similarities.append(sim_med)
            
            # Weighted combination emphasizing finest details
            weights = [0.7, 0.3]  # More weight on finest scale
            fine_reward = sum(w * s.item() for w, s in zip(weights, similarities))
            
            return fine_reward
    
    def get_combined_reward(self, f_mr_3d, f_ct_3d, warped_mr_features, flow_acc, 
                           epoch, total_epochs, smooth_val, mag_val, smooth_weight, mag_weight, sim_weight):
        """
        Combine coarse and fine rewards with curriculum annealing
        Early training: Focus on coarse global alignment (like DINO-Reg rigid step)
        Later training: Focus on fine local deformations
        """
        
        # Call coarse reward with the original unwarped f_mr_3d
        coarse_reward = self.compute_cls_alignment_reward(f_mr_3d, f_ct_3d)
        
        # The fine reward correctly uses the warped features
        fine_reward = self.compute_fine_alignment_reward(f_mr_3d, f_ct_3d, warped_mr_features)
        
        # Annealing: Start with coarse (global alignment), shift to fine (local details)
        warmup_epochs = 10 # Hold coarse_weight at 1.0 for this many epochs

        if epoch < warmup_epochs:
            coarse_weight = 1.0
        elif epoch < self.annealing_epochs:
            # Adjust progress to be from 0 to 1 within the annealing window
            progress = (epoch - warmup_epochs) / (self.annealing_epochs - warmup_epochs)
            cosine_val = 0.5 * (1 + np.cos(np.pi * progress))
            coarse_weight = self.coarse_weight_end + (self.coarse_weight_start - self.coarse_weight_end) * cosine_val
        else:
            coarse_weight = self.coarse_weight_end

        fine_weight = 1.0 - coarse_weight
        
        # Combine rewards
        alignment_reward = coarse_weight * coarse_reward + fine_weight * fine_reward
        
        # Total reward with regularization
        total_reward = 100 * (sim_weight * alignment_reward) - (smooth_weight * smooth_val) - (mag_weight * mag_val)
        
        return total_reward, coarse_reward, fine_reward, coarse_weight

class RewardNormalizer:
    """Calculates a running mean and standard deviation for rewards."""
    def __init__(self, num_inputs, clip=10.0):
        self.clip = clip
        self.n = 0
        self.mean = np.zeros(1)
        self.mean_diff = np.zeros(1)
        self.var = np.zeros(1)

    def update(self, reward):
        self.n += 1
        delta = reward - self.mean
        self.mean += delta / self.n
        self.mean_diff += delta * (reward - self.mean)
        self.var = self.mean_diff / self.n if self.n > 1 else np.square(self.mean)
    
    def normalize(self, rewards_tensor):
        # Detach, move to CPU, and convert to numpy for calculation
        rewards_np = rewards_tensor.detach().cpu().numpy()
        
        # Get the running standard deviation, avoid division by zero
        std = np.sqrt(self.var) + 1e-8
        
        # Normalize, clip to prevent extreme values, and convert back to tensor
        normalized_rewards = np.clip((rewards_np - self.mean) / std, -self.clip, self.clip)
        return torch.tensor(normalized_rewards, dtype=torch.float32, device=rewards_tensor.device)
    
def compute_smoothness_penalty(flow):
    """
    Computes L1 smoothness (total variation) of displacement field.
    flow: (B,3,D,H,W)
    """
    dx = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).mean()
    dy = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]).mean()
    dz = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]).mean()
    
    return dx + dy + dz

def compute_hierarchical_reward(fixed_feats, warped_feats, epoch, max_epochs,
                                smoothness_penalty, magnitude_penalty,
                                smooth_weight=0.5, mag_weight=2.0):
    """
    Computes a hierarchical reward that anneals from coarse (global) to fine (local) similarity.
    """
    # --- Fine-grained Reward (Local Similarity) ---
    # Your original multi_scale_reward is a good measure of fine-grained similarity.
    fine_similarity = multi_scale_reward(fixed_feats, warped_feats)

    # --- Coarse-grained Reward (Global Similarity) ---
    # We create a coarse representation by average pooling the feature maps.
    # This forces the agent to match the overall, low-frequency structure first.
    coarse_fixed = [F.adaptive_avg_pool3d(f, (1, 1, 1)) for f in fixed_feats]
    coarse_warped = [F.adaptive_avg_pool3d(f, (1, 1, 1)) for f in warped_feats]
    coarse_similarity = multi_scale_reward(coarse_fixed, coarse_warped)

    # --- Annealing Schedule ---
    # Start by focusing on the coarse reward, then gradually shift focus to the fine reward.
    # We use a cosine schedule for a smooth transition.
    progress = epoch / max_epochs
    coarse_weight = 0.5 * (1 + math.cos(math.pi * progress)) # Starts at 1.0, ends at 0.0
    fine_weight = 1 - coarse_weight                         # Starts at 0.0, ends at 1.0
    
    # --- Final Combined Reward ---
    similarity_reward = (coarse_weight * coarse_similarity) + (fine_weight * fine_similarity)
    
    total_reward = similarity_reward - (smooth_weight * smoothness_penalty) - (mag_weight * magnitude_penalty)
    
    return total_reward, similarity_reward, coarse_similarity, fine_similarity

def local_cosine(fixed, moving, win=9):
    """Local cosine similarity over patches."""
    unfold = nn.Unfold(kernel_size=win, stride=1, padding=win//2)
    # flatten patches
    f_p = unfold(fixed).transpose(1, 2)  # (B, N_patches, C*win^2)
    m_p = unfold(moving).transpose(1, 2)

    f_p = F.normalize(f_p, dim=-1)
    m_p = F.normalize(m_p, dim=-1)

    sim = (f_p * m_p).sum(-1)  # cosine similarity per patch
    return sim.mean()

def multi_scale_reward(fixed_feats, warped_feats):
    sims = []
    for f_fix, f_warp in zip(fixed_feats, warped_feats):
        f_fix = F.normalize(f_fix,dim=1)
        f_warp = F.normalize(f_warp,dim=1)
        sims.append((f_fix*f_warp).sum(1).mean())
    return torch.stack(sims).mean()

def compute_image_gradients(img):
    """
    Compute gradients along x, y, z for 3D images.
    img: (B, C, D, H, W)
    Returns: (B, C, 3, D, H, W)  [dx, dy, dz]
    """
    dz = img[:, :, 2:, :, :] - img[:, :, :-2, :, :]
    dy = img[:, :, :, 2:, :] - img[:, :, :, :-2, :]
    dx = img[:, :, :, :, 2:] - img[:, :, :, :, :-2]

    # pad to keep same size
    dz = F.pad(dz, (0,0,0,0,1,1))
    dy = F.pad(dy, (0,0,1,1,0,0))
    dx = F.pad(dx, (1,1,0,0,0,0))

    grads = torch.stack([dx, dy, dz], dim=2)
    return grads

def ngf_loss(fixed, moving_warped, eps=1e-5):
    """
    NGF loss: encourages gradient directions to align
    fixed: (B,1,D,H,W) fixed image
    moving_warped: (B,1,D,H,W) warped moving image
    """
    grad_f = compute_image_gradients(fixed)
    grad_m = compute_image_gradients(moving_warped)

    # normalize gradients
    grad_f = grad_f / (torch.norm(grad_f, dim=2, keepdim=True) + eps)
    grad_m = grad_m / (torch.norm(grad_m, dim=2, keepdim=True) + eps)

    # 1 - cosine similarity between gradients
    sim = (grad_f * grad_m).sum(2)  # (B,C,D,H,W)
    loss = 1 - sim.mean()
    return loss

# Alternative: Feature-Image Consistency Check (Optional Addition)
class FeatureImageConsistencyReward:
    def __init__(self, device, consistency_weight=0.1):
        self.device = device
        self.consistency_weight = consistency_weight
        
    def compute_consistency_reward(self, mr_img, ct_img, warped_mr_img, 
                                 f_mr_features, f_ct_features, warped_mr_features):
        """
        Lightweight consistency check between image-level and feature-level similarities
        Uses downsampled images to reduce computational cost
        """
        with torch.no_grad():
            # Downsample images for efficiency (e.g., to 64^3)
            target_size = (64, 64, 64)
            mr_small = F.interpolate(mr_img, size=target_size, mode='trilinear')
            ct_small = F.interpolate(ct_img, size=target_size, mode='trilinear')  
            warped_mr_small = F.interpolate(warped_mr_img, size=target_size, mode='trilinear')
            
            # Compute normalized cross-correlation on downsampled images
            def ncc_small(img1, img2):
                img1_flat = img1.flatten()
                img2_flat = img2.flatten()
                img1_norm = (img1_flat - img1_flat.mean()) / (img1_flat.std() + 1e-8)
                img2_norm = (img2_flat - img2_flat.mean()) / (img2_flat.std() + 1e-8)
                return torch.mean(img1_norm * img2_norm)
            
            image_similarity = ncc_small(warped_mr_small, ct_small)
            
            # Compute feature similarity  
            feature_similarity = F.cosine_similarity(
                warped_mr_features.flatten(), 
                f_ct_features.flatten(), 
                dim=0
            )
            
            # Consistency reward: penalize when image and feature similarities disagree
            consistency = 1.0 - abs(image_similarity.item() - feature_similarity.item())
            
            return self.consistency_weight * consistency

def normalize_advantages(advantages, eps=1e-8):
    """
    Normalize advantages for more stable training (V-Learn paper technique)
    
    Args:
        advantages: Tensor of advantages [batch_size, ...]
        eps: Small epsilon for numerical stability
    
    Returns:
        Normalized advantages with zero mean and unit variance
    """
    if advantages.numel() == 1:
        # Single advantage, return as-is
        return advantages
    
    # Flatten advantages for statistics calculation
    flat_advantages = advantages.view(-1)
    
    # Calculate mean and std
    mean = flat_advantages.mean()
    std = flat_advantages.std() + eps
    
    # Normalize
    normalized = (advantages - mean) / std
    
    # Additional safety check
    if not torch.isfinite(normalized).all():
        print("WARNING: NaN/Inf in advantage normalization. Returning a detached clone of original advantages to prevent leaks.")
        # Return a new tensor with no computation history
        return advantages.clone().detach()
    
    return normalized

class DualChannelRewardSystem:
    def __init__(self, device):
        self.device = device
        # Use simple running stats for normalization, not EMA, for better stability
        self.reward_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}

    def compute_cls_alignment_reward(self, f_mr_3d, f_ct_3d):
        """
        Compute coarse alignment reward using global features. Unchanged.
        """
        with torch.no_grad():
            mr_global = F.adaptive_avg_pool3d(f_mr_3d.unsqueeze(0), (1, 1, 1)).squeeze()
            ct_global = F.adaptive_avg_pool3d(f_ct_3d.unsqueeze(0), (1, 1, 1)).squeeze()
            return F.cosine_similarity(mr_global, ct_global, dim=0).item()

    def mind_descriptor(self, img, radius=1, dilation=1):
        """
        NEW: A robust, differentiable MIND-SSD reward. This is the key fix.
        Lower MIND-SSD (better alignment) should result in a higher reward.
        """
        # This function expects a 5D tensor B, C, D, H, W
        if img.dim() != 5:
            raise ValueError("Input to mind_descriptor must be a 5D tensor")

        # Pad the input tensor
        padded = F.pad(img, (radius, radius, radius, radius, radius, radius), mode='replicate')

        out = []
        # Loop through all shifts in the 3D neighborhood
        for dz in range(-radius, radius + 1, dilation):
            for dy in range(-radius, radius + 1, dilation):
                for dx in range(-radius, radius + 1, dilation):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue

                    # --- THE CORRECTED SLICING LOGIC ---
                    # This uses negative indexing to be more robust
                    start_z, end_z = radius + dz, -(radius - dz) if (radius - dz) != 0 else None
                    start_y, end_y = radius + dy, -(radius - dy) if (radius - dy) != 0 else None
                    start_x, end_x = radius + dx, -(radius - dx) if (radius - dx) != 0 else None
                    
                    shifted = padded[:, :, start_z:end_z, start_y:end_y, start_x:end_x]
                    out.append(shifted)
        
        # Stack the shifted tensors to create the descriptor channels
        out = torch.stack(out, dim=1)  # B, N, C, D, H, W
        
        # Compute the variance-based descriptor
        dist = torch.mean((img.unsqueeze(1) - out) ** 2, dim=1) # Variance across shifts
        dist_var = torch.mean((dist - torch.mean(dist, dim=1, keepdim=True))**2, dim=1) # Variance of variances
        
        # Clamp the variance to prevent division by zero for stable gradients
        dist_var = dist_var.clamp(min=1e-5)
        
        # Final MIND descriptor
        mind = torch.exp(-dist / dist_var)
        return mind
        
    def compute_mind_ssd_reward(self, warped_features, fixed_features):
        """
        NEW: A robust, differentiable MIND-SSD reward.
        This version maps the loss to a reward in the range [0, 1].
        """

        with torch.no_grad():
            mind_warped = self.mind_descriptor(warped_features.unsqueeze(0))
            mind_fixed = self.mind_descriptor(fixed_features.unsqueeze(0))

            # Sum of Squared Differences between the robust MIND features
            mind_ssd_loss = ((mind_warped - mind_fixed) ** 2).mean()

            # --- THE FIX: Map the loss to a reward ---
            # A lower loss (better alignment) will result in a reward closer to 1.0
            # A higher loss (worse alignment) will result in a reward closer to 0.0
            # The 'temperature' parameter (e.g., 0.1) controls the sharpness of the reward.
            temperature = 0.1 
            reward = torch.exp(-mind_ssd_loss / temperature)
            
            return reward.item()

    def get_combined_reward(self, f_mr_3d, f_ct_3d, warped_mr_features, flow_acc,
                           epoch, total_epochs, smooth_val, mag_val, smooth_weight, mag_weight, sim_weight):
        
        coarse_reward = self.compute_cls_alignment_reward(f_mr_3d, f_ct_3d)
        
        # Use the new, stable MIND-SSD reward for fine-tuning
        fine_reward = self.compute_mind_ssd_reward(warped_mr_features, f_ct_3d)
        
        # Use fixed weights. The agent is rewarded for both global and local alignment.
        coarse_weight = 0.3
        fine_weight = 0.7 # Emphasize the more informative MIND-SSD reward

        alignment_reward = coarse_weight * coarse_reward + fine_weight * fine_reward
        
        # Scale rewards to positive range and balance components better
        # Scale alignment reward to [0, 100] range for stability
        scaled_alignment = 100 * max(0, alignment_reward)
        
        # Much gentler penalties - let the agent learn alignment first
        penalty_scale = 0.1  # Very light penalties during learning
        total_reward = scaled_alignment - penalty_scale * (smooth_weight * smooth_val + mag_weight * mag_val)
        
        # Ensure reward stays positive to encourage exploration
        total_reward = max(total_reward, 0.1)
        
        return total_reward, coarse_reward, fine_reward, coarse_weight
    
class SimplifiedSSDRewardSystem:
    def __init__(self, device):
        self.device = device

    def get_reward(self, warped_features, fixed_features, flow_field, smoothness_weight, magnitude_weight):
        """
        A simple, robust reward system based on feature similarity (SSD) and regularization.
        """
        with torch.no_grad():
            # 1. Similarity Reward (lower SSD is better, so we negate it for reward)
            similarity_loss = ((warped_features - fixed_features)**2).mean()
            similarity_reward = -similarity_loss.item()

            # 2. Smoothness Penalty
            dz = flow_field[:, :, 1:, :, :] - flow_field[:, :, :-1, :, :]
            dy = flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]
            dx = flow_field[:, :, :, :, 1:] - flow_field[:, :, :, :, :-1]
            smoothness_penalty = (dz**2).mean() + (dy**2).mean() + (dx**2).mean()
            
            # 3. Magnitude Penalty
            magnitude_penalty = (flow_field**2).mean()

            # Combine the components with a high weight on similarity
            total_reward = (1000 * similarity_reward) - \
                           (smoothness_weight * smoothness_penalty.item()) - \
                           (magnitude_weight * magnitude_penalty.item())

            return total_reward
        
class MindSSDRewardSystem:
    def __init__(self, device):
        self.device = device

    def mind_descriptor(self, img, radius=1, dilation=1):
        """
        Calculates the MIND descriptor for a given feature map.
        This version is corrected to prevent crashes and instability.
        """
        # Expects a 5D tensor B, C, D, H, W
        B, C, D, H, W = img.shape
        
        # Pad the input tensor
        padded = F.pad(img, (radius, radius, radius, radius, radius, radius), mode='replicate')

        out = []
        # Loop through all shifts in the 3D neighborhood
        for dz in range(-radius, radius + 1, dilation):
            for dy in range(-radius, radius + 1, dilation):
                for dx in range(-radius, radius + 1, dilation):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    # Correct, robust slicing
                    shifted = padded[:, :, radius + dz : radius + dz + D, 
                                     radius + dy : radius + dy + H, 
                                     radius + dx : radius + dx + W]
                    out.append(shifted)
        
        out = torch.stack(out, dim=1)  # B, N, C, D, H, W
        
        dist = torch.mean((img.unsqueeze(1) - out) ** 2, dim=1)
        dist_var = torch.mean((dist - torch.mean(dist, dim=1, keepdim=True))**2, dim=1)
        
        # Clamp the variance to prevent division by zero for stable gradients
        dist_var = dist_var.clamp(min=1e-5)
        
        mind = torch.exp(-dist / dist_var)
        return mind

    # In your MindSSDRewardSystem class
    def get_reward(self, warped_features, fixed_features, flow_field, 
                smoothness_weight, magnitude_weight, similarity_weight):
        """
        A simple, robust, cost-based reward system.
        """
        with torch.no_grad():
            mind_warped = self.mind_descriptor(warped_features.unsqueeze(0))
            mind_fixed = self.mind_descriptor(fixed_features.unsqueeze(0))
            
            # 1. Calculate the three separate "costs"
            similarity_cost = ((mind_warped - mind_fixed)**2).mean().item()
            
            dz = flow_field[:, :, 1:, :, :] - flow_field[:, :, :-1, :, :]
            dy = flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]
            dx = flow_field[:, :, :, :, 1:] - flow_field[:, :, :, :, :-1]
            smoothness_cost = (dz.abs()).mean().item() + (dy.abs()).mean().item() + (dx.abs()).mean().item()
            
            magnitude_cost = (flow_field.abs()).mean().item()
            
            # 2. Calculate the total weighted cost
            total_cost = (similarity_weight * similarity_cost) + \
                        (smoothness_weight * smoothness_cost) + \
                        (magnitude_weight * magnitude_cost)
                        
            # 3. The reward is the negative of the total cost. This is a stable signal.
            # The agent's reward is *only* the image similarity.
            # We want to MAXIMIZE similarity. ncc_similarity is [B] (higher is better).
            # We scale it by the similarity_weight.
            total_reward = ncc_similarity * similarity_weight

            # We still return the TENSOR costs for the decoder loss and logging
            return total_reward, similarity_cost, smoothness_cost, magnitude_cost
        
class StableNCCRewardSystem:
    """
    A much simpler, more stable reward system using Normalized Cross-Correlation.
    This provides clearer gradients for deformable registration learning.
    """
    def __init__(self, device):
        self.device = device
        
    def normalized_cross_correlation(self, img1, img2):
        """Compute normalized cross-correlation between two images."""
        # Flatten the images
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        
        # Normalize to zero mean, unit variance
        img1_norm = (img1_flat - img1_flat.mean()) / (img1_flat.std() + 1e-8)
        img2_norm = (img2_flat - img2_flat.mean()) / (img2_flat.std() + 1e-8)
        
        # Compute correlation
        ncc = torch.mean(img1_norm * img2_norm)
        return ncc.item()
    
    def get_reward(self, warped_features, fixed_features, flow_field, 
                   smoothness_weight, magnitude_weight, similarity_weight):
        """Simple, stable reward based on NCC similarity."""
        with torch.no_grad():
            # Compute NCC similarity (range [-1, 1], higher is better)
            ncc_similarity = self.normalized_cross_correlation(warped_features, fixed_features)
            
            # Convert to positive reward (range [0, 2])
            similarity_reward = (ncc_similarity + 1.0) * 50.0  # Scale to [0, 100]
            
            # Simple regularization penalties
            flow_magnitude = torch.norm(flow_field, p=2, dim=1).mean().item()
            
            # Gradient-based smoothness penalty (more meaningful than finite differences)
            dz = flow_field[:, :, 1:, :, :] - flow_field[:, :, :-1, :, :]
            dy = flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]
            dx = flow_field[:, :, :, :, 1:] - flow_field[:, :, :, :, :-1]
            smoothness_cost = (dx.abs().mean() + dy.abs().mean() + dz.abs().mean()).item()
            
            # Magnitude penalty
            magnitude_cost = flow_magnitude
            
            # Adaptive penalty scaling
            if flow_magnitude < 0.5:  # Too little movement - reduce penalties
                penalty_scale = 0.1
            elif flow_magnitude > 15.0:  # Too much movement - increase penalties
                penalty_scale = 3.0
            else:  # Normal range - standard penalties
                penalty_scale = 1.0
            
            smoothness_penalty = smoothness_weight * smoothness_cost * penalty_scale
            magnitude_penalty = magnitude_weight * magnitude_cost * penalty_scale
            
            # Final reward
            total_reward = similarity_reward - smoothness_penalty - magnitude_penalty
            
            return total_reward, similarity_reward, smoothness_cost, magnitude_cost

class StableRegistrationReward:
    """
    Simplified, stable reward system focused specifically on deformable registration.
    Uses normalized cross-correlation which provides clearer learning signals than MIND.
    """
    def __init__(self, device):
        self.device = device
        
    def normalized_cross_correlation(self, img1, img2):
        """Compute normalized cross-correlation - excellent for registration."""
        # Flatten and normalize
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        
        # Zero mean, unit variance normalization
        img1_norm = (img1_flat - img1_flat.mean()) / (img1_flat.std() + 1e-8)
        img2_norm = (img2_flat - img2_flat.mean()) / (img2_flat.std() + 1e-8)
        
        # Correlation coefficient
        ncc = torch.mean(img1_norm * img2_norm)
        return ncc
    
    def get_reward(self, warped_features, fixed_features, flow_field, 
                   smoothness_weight, magnitude_weight, similarity_weight):
        """Simple, stable reward focused on registration quality with enhanced scaling."""
        with torch.no_grad():
            # Primary reward: NCC similarity (range [-1, 1], higher is better)
            ncc_similarity = self.normalized_cross_correlation(warped_features, fixed_features)
            
            # Enhanced similarity reward with better scaling
            # Transform [-1,1] to [0,200] but with exponential scaling for better discrimination
            if ncc_similarity > 0:
                # Exponential scaling for positive correlations to reward good alignments more
                similarity_reward = 100.0 * (1.0 + ncc_similarity) * (1.0 + ncc_similarity * 0.5)
            else:
                # Linear penalty for negative correlations
                similarity_reward = 100.0 * (1.0 + ncc_similarity)
            
            # Simple L2 magnitude penalty
            magnitude_penalty = torch.mean(flow_field ** 2).item()
            
            # Simple smoothness penalty using total variation
            dx = torch.mean((flow_field[:, :, :, :, 1:] - flow_field[:, :, :, :, :-1]) ** 2).item()
            dy = torch.mean((flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]) ** 2).item()
            dz = torch.mean((flow_field[:, :, 1:, :, :] - flow_field[:, :, :-1, :, :]) ** 2).item()
            smoothness_penalty = dx + dy + dz
            
            # Adaptive penalty scaling based on flow magnitude for stability
            flow_magnitude = torch.mean(torch.norm(flow_field, p=2, dim=1)).item()
            penalty_scale = 1.0
            if flow_magnitude < 0.5:  # Too little movement
                penalty_scale = 0.05  # Very light penalties to encourage movement
            elif flow_magnitude > 12.0:  # Too much movement
                penalty_scale = 2.0   # Stronger penalties
            elif flow_magnitude > 8.0:
                penalty_scale = 1.5   # Moderate penalties
                
            # Final reward calculation with clearer scaling
            total_reward = similarity_reward.item() - \
                          (smoothness_weight * smoothness_penalty * penalty_scale) - \
                          (magnitude_weight * magnitude_penalty * penalty_scale)
            
            return total_reward, similarity_reward.item(), smoothness_penalty, magnitude_penalty

class RawImageMINDSystem:
    """
    A reward system that applies the MIND descriptor directly to raw image volumes.
    """
    def __init__(self, device):
        self.device = device

    def mind_descriptor(self, img, radius=1, dilation=1):
        # This helper function handles both 3D and 5D tensors for raw images
        # Handle different input dimensions for raw images vs DINO features
        if img.dim() == 3:  # Raw image case: [D, H, W]
            # Add batch and channel dimensions: [1, 1, D, H, W]
            img = img.unsqueeze(0).unsqueeze(0)
        elif img.dim() == 4:  # Raw image case: [C, D, H, W] or [B, D, H, W]
            if img.shape[0] <= 4:  # Likely [B, D, H, W] where B is small
                img = img.unsqueeze(1)  # Add channel dim: [B, 1, D, H, W]
            else:  # Likely [C, D, H, W] where C is large (like 64 for features)
                img = img.unsqueeze(0)  # Add batch dim: [1, C, D, H, W]
        elif img.dim() != 5:
            raise ValueError(f"Unsupported tensor dimension: {img.dim()}. Expected 3, 4, or 5.")
        
        B, C, D, H, W = img.shape
        padded = F.pad(img, (radius,)*6, mode='replicate')
        out = []
        for dz in range(-radius, radius + 1, dilation):
            for dy in range(-radius, radius + 1, dilation):
                for dx in range(-radius, radius + 1, dilation):
                    if dx == 0 and dy == 0 and dz == 0: continue
                    shifted = padded[:, :, radius+dz:radius+dz+D, radius+dy:radius+dy+H, radius+dx:radius+dx+W]
                    out.append(shifted)
        out = torch.stack(out, dim=1)
        dist = torch.mean((img.unsqueeze(1) - out) ** 2, dim=2) # Mean over channel dim
        dist_var = torch.mean((dist - torch.mean(dist, dim=1, keepdim=True))**2, dim=1, keepdim=True)
        mind = torch.exp(-dist / (dist_var.clamp(min=1e-5)))
        return mind

    # --- START FIX: Add missing arguments to the function signature ---
    def get_reward(self, warped_features, fixed_features, flow_field, 
                   smoothness_weight, magnitude_weight, similarity_weight,
                   baseline_features=None, epoch=0, max_epochs=150):
    # --- END FIX ---
    
        # In this context, "warped_features" IS the warped MR image,
        # and "fixed_features" IS the fixed CT image.
        with torch.no_grad():
            # Apply the MIND descriptor to the raw image volumes
            mind_warped = self.mind_descriptor(warped_features)
            mind_fixed = self.mind_descriptor(fixed_features)
            
            # Calculate the costs
            # --- START FIX: Remove .item() ---
            similarity_cost = ((mind_warped - mind_fixed)**2).mean()
            
            dz = flow_field[:, :, 1:, :, :] - flow_field[:, :, :-1, :, :]
            dy = flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]
            dx = flow_field[:, :, :, :, 1:] - flow_field[:, :, :, :, :-1]
            smoothness_cost = (dz**2).mean() + (dy**2).mean() + (dx**2).mean()
            magnitude_cost = (flow_field**2).mean()
            
            # NEW: Better reward scaling to prevent saturation
            # Use logarithmic scaling for similarity to prevent extreme values
            similarity_reward = 100.0 * torch.exp(-similarity_cost * 0.1)
            
            # Apply adaptive penalty scaling based on flow magnitude
            flow_magnitude = torch.norm(flow_field, p=2, dim=1).mean()
            # --- END FIX ---
            
            if flow_magnitude > 10.0:  # High flow - increase penalties
                penalty_scale = 2.0
            else:  # Normal flow - standard penalties
                penalty_scale = 1.0

            smoothness_penalty = smoothness_weight * smoothness_cost * penalty_scale
            magnitude_penalty = magnitude_weight * magnitude_cost * penalty_scale
            
            total_reward = similarity_reward - smoothness_penalty - magnitude_penalty
            
            # Ensure reward stays in reasonable range [-100, 100]
            total_reward = torch.clamp(total_reward, -100.0, 100.0)
            
            similarity_reward_for_logging = similarity_reward
            
            # Return TENSORS, not floats
            return total_reward, similarity_reward_for_logging, smoothness_cost, magnitude_cost
        
class EnhancedStableReward:
    """
    A robust, cost-based reward system.
    Returns TENSORS, not floats.
    """
    def __init__(self, device):
        self.device = device
        
    def normalized_cross_correlation(self, img1, img2):
        """
        Batch-aware and stable NCC computation.
        img1, img2: [B, C, D, H, W]
        Returns: [B] tensor of NCC scores (similarity, higher is better)
        """
        B = img1.shape[0]
        img1_flat = img1.view(B, -1)
        img2_flat = img2.view(B, -1)
        
        img1_mean = img1_flat.mean(dim=-1, keepdim=True)
        img2_mean = img2_flat.mean(dim=-1, keepdim=True)
        
        epsilon = 1e-8
        img1_std = img1_flat.std(dim=-1, keepdim=True) + epsilon
        img2_std = img2_flat.std(dim=-1, keepdim=True) + epsilon

        img1_norm = (img1_flat - img1_mean) / img1_std
        img2_norm = (img2_flat - img2_mean) / img2_std
        
        ncc = torch.mean(img1_norm * img2_norm, dim=-1)
        
        return torch.clamp(ncc, -1.0, 1.0) # Return similarity, not cost
    
    def multi_scale_similarity(self, warped_features, fixed_features):
        sim_full = self.normalized_cross_correlation(warped_features, fixed_features)
        
        try:
            warped_small = F.avg_pool3d(warped_features, 2, 2)
            fixed_small = F.avg_pool3d(fixed_features, 2, 2)
            sim_small = self.normalized_cross_correlation(warped_small, fixed_small)
            final_sim = 0.7 * sim_full + 0.3 * sim_small
        except Exception:
            final_sim = sim_full
            
        return final_sim # [B] tensor
        
    def get_reward(self, warped_features, fixed_features, flow_field, 
                   smoothness_weight, magnitude_weight, similarity_weight):
        
        ncc_similarity = self.multi_scale_similarity(warped_features, fixed_features)
        similarity_cost = 1.0 - ncc_similarity
        
        # Compute penalties
        dx = torch.mean((flow_field[:, :, :, :, 1:] - flow_field[:, :, :, :, :-1]).abs(), dim=(1, 2, 3, 4))
        dy = torch.mean((flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]).abs(), dim=(1, 2, 3, 4))
        dz = torch.mean((flow_field[:, :, 1:, :, :] - flow_field[:, :, :-1, :, :]).abs(), dim=(1, 2, 3, 4))
        smoothness_cost = (dx + dy + dz)
        magnitude_cost = torch.mean(flow_field ** 2, dim=(1, 2, 3, 4))
        
        # FIX: Scale the reward to a reasonable range [0, 100]
        # Positive rewards for good alignment
        similarity_reward = 100.0 * ncc_similarity  # Range [-100, 100]
        
        # Light penalties that don't overwhelm the similarity signal
        total_reward = similarity_reward - \
                       (smoothness_weight * smoothness_cost) - \
                       (magnitude_weight * magnitude_cost)
        
        return total_reward, similarity_cost, smoothness_cost, magnitude_cost
    
class ImprovementBasedReward:
    """
    Rewards the agent for IMPROVING alignment, not just achieving good alignment.
    This prevents the "do nothing" collapse.
    """
    def __init__(self, device):
        self.device = device
        
    def normalized_cross_correlation(self, img1, img2):
        """Batch-aware NCC computation"""
        B = img1.shape[0]
        img1_flat = img1.view(B, -1)
        img2_flat = img2.view(B, -1)
        
        img1_mean = img1_flat.mean(dim=-1, keepdim=True)
        img2_mean = img2_flat.mean(dim=-1, keepdim=True)
        
        epsilon = 1e-8
        img1_std = img1_flat.std(dim=-1, keepdim=True) + epsilon
        img2_std = img2_flat.std(dim=-1, keepdim=True) + epsilon

        img1_norm = (img1_flat - img1_mean) / img1_std
        img2_norm = (img2_flat - img2_mean) / img2_std
        
        ncc = torch.mean(img1_norm * img2_norm, dim=-1)
        
        return torch.clamp(ncc, -1.0, 1.0)
    
    def get_reward(self, warped_features, fixed_features, flow_field, 
                   smoothness_weight, magnitude_weight, similarity_weight,
                   baseline_features=None):
        """
        baseline_features: The UNWARPED moving features (for computing improvement)
        """
        # Current alignment quality
        ncc_current = self.normalized_cross_correlation(warped_features, fixed_features)
        
        # Baseline alignment (before any deformation)
        if baseline_features is not None:
            ncc_baseline = self.normalized_cross_correlation(baseline_features, fixed_features)
            # Reward = improvement from baseline
            improvement = ncc_current - ncc_baseline
            # Scale to reasonable range [-100, 100]
            similarity_reward = 100.0 * improvement
        else:
            # Fallback: use absolute similarity (but this leads to the collapse)
            similarity_reward = 100.0 * ncc_current
        
        # Compute penalties
        dx = torch.mean((flow_field[:, :, :, :, 1:] - flow_field[:, :, :, :, :-1]).abs(), dim=(1, 2, 3, 4))
        dy = torch.mean((flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]).abs(), dim=(1, 2, 3, 4))
        dz = torch.mean((flow_field[:, :, 1:, :, :] - flow_field[:, :, :-1, :, :]).abs(), dim=(1, 2, 3, 4))
        smoothness_cost = (dx + dy + dz)
        magnitude_cost = torch.mean(flow_field ** 2, dim=(1, 2, 3, 4))
        
        # Similarity cost for logging (1 - NCC)
        similarity_cost = 1.0 - ncc_current
        
        # Total reward: improvement - penalties
        total_reward = similarity_reward - \
                       (smoothness_weight * smoothness_cost) - \
                       (magnitude_weight * magnitude_cost)
        
        return total_reward, similarity_cost, smoothness_cost, magnitude_cost
    
class HybridRegistrationReward:
    """
    CRITICAL FIX: Scale rewards to match DINO-Reg's actual similarity values
    """
    def __init__(self, device):
        self.device = device
        
    def normalized_cross_correlation(self, img1, img2):
        """Standard NCC - returns values in [-1, 1]"""
        B = img1.shape[0]
        img1_flat = img1.view(B, -1)
        img2_flat = img2.view(B, -1)
        
        img1_mean = img1_flat.mean(dim=-1, keepdim=True)
        img2_mean = img2_flat.mean(dim=-1, keepdim=True)
        
        epsilon = 1e-8
        img1_std = img1_flat.std(dim=-1, keepdim=True) + epsilon
        img2_std = img2_flat.std(dim=-1, keepdim=True) + epsilon

        img1_norm = (img1_flat - img1_mean) / img1_std
        img2_norm = (img2_flat - img2_mean) / img2_std
        
        ncc = torch.mean(img1_norm * img2_norm, dim=-1)
        return torch.clamp(ncc, -1.0, 1.0)
    
    def get_reward(self, warped_features, fixed_features, flow_field, 
                   smoothness_weight, magnitude_weight, similarity_weight,
                   baseline_features=None, epoch=0, max_epochs=150):
        """
        CRITICAL: Return rewards that match the value range critic expects
        
        DINO-Reg achieves NCC ~0.7-0.8 for good alignment
        Baseline (no registration) is ~0.3-0.4
        So improvement is ~0.3-0.5
        
        Scale this to [-50, +50] range for RL
        """
        
        # Compute current similarity
        ncc_current = self.normalized_cross_correlation(warped_features, fixed_features)
        
        if baseline_features is not None:
            ncc_baseline = self.normalized_cross_correlation(baseline_features, fixed_features)
            improvement = ncc_current - ncc_baseline
            
            # CRITICAL: Scale to reasonable range
            # improvement ranges from -0.5 to +0.5 typically
            # Scale to [-50, +50]
            similarity_reward = improvement * 100.0
        else:
            # No baseline: assume 0.4 as typical baseline NCC
            improvement = ncc_current - 0.4
            similarity_reward = improvement * 100.0
        
        # Penalties (keep small)
        dx = torch.mean((flow_field[:, :, :, :, 1:] - flow_field[:, :, :, :, :-1]) ** 2, dim=(1, 2, 3, 4))
        dy = torch.mean((flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]) ** 2, dim=(1, 2, 3, 4))
        dz = torch.mean((flow_field[:, :, 1:, :, :] - flow_field[:, :, :-1, :, :]) ** 2, dim=(1, 2, 3, 4))
        smoothness_cost = (dx + dy + dz)
        
        magnitude_cost = torch.mean(flow_field ** 2, dim=(1, 2, 3, 4))
        
        # Total reward: improvement - small penalties
        total_reward = similarity_reward - \
                       (smoothness_weight * smoothness_cost) - \
                       (magnitude_weight * magnitude_cost)
        
        # For logging
        similarity_cost = 1.0 - ncc_current
        
        # Print debug info occasionally
        if torch.rand(1).item() < 0.01:  # 1% of the time
            print(f"  [Reward Debug] NCC: {ncc_current.mean():.3f}, "
                  f"Improvement: {improvement.mean():.3f}, "
                  f"Reward: {total_reward.mean():.1f}")
        
        return total_reward, similarity_cost, smoothness_cost, magnitude_cost

class GradientFieldReward:
    """
    CRITICAL INSIGHT: DINO features have high NCC even when misaligned.
    Instead, use GRADIENT ALIGNMENT which correlates much better with Dice.
    
    This is similar to DINO-Reg using MIND features (gradient-based).
    """
    def __init__(self, device):
        self.device = device
        
    def compute_image_gradients_3d(self, volume):
        """
        Compute 3D gradients (similar to MIND descriptor)
        These are much more sensitive to misalignment than raw intensity.
        """
        # Sobel-like gradients in 3D
        # X direction
        dx = volume[:, :, :, :, 2:] - volume[:, :, :, :, :-2]
        dx = F.pad(dx, (1, 1, 0, 0, 0, 0), mode='replicate')
        
        # Y direction  
        dy = volume[:, :, :, 2:, :] - volume[:, :, :, :-2, :]
        dy = F.pad(dy, (0, 0, 1, 1, 0, 0), mode='replicate')
        
        # Z direction
        dz = volume[:, :, 2:, :, :] - volume[:, :, :-2, :, :]
        dz = F.pad(dz, (0, 0, 0, 0, 1, 1), mode='replicate')
        
        # Gradient magnitude (more stable than individual components)
        grad_mag = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)
        
        return grad_mag
    
    def normalized_cross_correlation(self, img1, img2):
        """Standard NCC"""
        B = img1.shape[0]
        img1_flat = img1.view(B, -1)
        img2_flat = img2.view(B, -1)
        
        img1_mean = img1_flat.mean(dim=-1, keepdim=True)
        img2_mean = img2_flat.mean(dim=-1, keepdim=True)
        
        epsilon = 1e-8
        img1_std = img1_flat.std(dim=-1, keepdim=True) + epsilon
        img2_std = img2_flat.std(dim=-1, keepdim=True) + epsilon

        img1_norm = (img1_flat - img1_mean) / img1_std
        img2_norm = (img2_flat - img2_mean) / img2_std
        
        ncc = torch.mean(img1_norm * img2_norm, dim=-1)
        return torch.clamp(ncc, -1.0, 1.0)
    
    def get_reward(self, warped_features, fixed_features, flow_field, 
                   smoothness_weight, magnitude_weight, similarity_weight,
                   baseline_features=None, epoch=0, max_epochs=150):
        """
        MULTI-SCALE REWARD:
        1. Gradient alignment (most sensitive)
        2. Feature NCC (medium sensitivity)  
        3. Coarse NCC (least sensitive)
        
        This creates a stronger learning signal than NCC alone.
        """
        
        # 1. GRADIENT ALIGNMENT REWARD (CRITICAL!)
        # Compute gradients of features
        grad_warped = self.compute_image_gradients_3d(warped_features)
        grad_fixed = self.compute_image_gradients_3d(fixed_features)
        
        # NCC of gradients (much more sensitive than NCC of features)
        grad_ncc = self.normalized_cross_correlation(grad_warped, grad_fixed)
        
        if baseline_features is not None:
            grad_baseline = self.compute_image_gradients_3d(baseline_features)
            grad_ncc_baseline = self.normalized_cross_correlation(grad_baseline, grad_fixed)
            grad_improvement = grad_ncc - grad_ncc_baseline
        else:
            # Assume poor baseline gradient alignment
            grad_improvement = grad_ncc - 0.3
        
        # Scale gradient improvement (this is the PRIMARY signal)
        gradient_reward = grad_improvement * 200.0  # High weight
        
        # 2. FEATURE NCC REWARD (Secondary signal)
        feature_ncc = self.normalized_cross_correlation(warped_features, fixed_features)
        
        if baseline_features is not None:
            feature_ncc_baseline = self.normalized_cross_correlation(baseline_features, fixed_features)
            feature_improvement = feature_ncc - feature_ncc_baseline
        else:
            feature_improvement = feature_ncc - 0.65  # Typical baseline
        
        feature_reward = feature_improvement * 100.0
        
        # 3. COARSE ALIGNMENT REWARD (Tertiary signal for stability)
        try:
            # Downsample to 16^3 for coarse check
            warped_coarse = F.avg_pool3d(warped_features, kernel_size=2, stride=2)
            fixed_coarse = F.avg_pool3d(fixed_features, kernel_size=2, stride=2)
            coarse_ncc = self.normalized_cross_correlation(warped_coarse, fixed_coarse)
            
            if baseline_features is not None:
                baseline_coarse = F.avg_pool3d(baseline_features, kernel_size=2, stride=2)
                coarse_ncc_baseline = self.normalized_cross_correlation(baseline_coarse, fixed_coarse)
                coarse_improvement = coarse_ncc - coarse_ncc_baseline
            else:
                coarse_improvement = coarse_ncc - 0.60
            
            coarse_reward = coarse_improvement * 50.0
        except:
            coarse_reward = torch.zeros_like(feature_reward)
        
        # COMBINED SIMILARITY REWARD (weighted combination)
        # Gradient alignment is most important for Dice improvement
        similarity_reward = (
            0.6 * gradient_reward +   # PRIMARY: Gradient alignment
            0.3 * feature_reward +    # Secondary: Feature NCC
            0.1 * coarse_reward       # Tertiary: Coarse alignment
        )
        
        # 4. PENALTIES (Adaptive based on flow magnitude)
        flow_magnitude = torch.mean(torch.norm(flow_field, p=2, dim=1), dim=(1, 2, 3))
        
        # Smoothness (L2)
        dx = torch.mean((flow_field[:, :, :, :, 1:] - flow_field[:, :, :, :, :-1]) ** 2, dim=(1, 2, 3, 4))
        dy = torch.mean((flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]) ** 2, dim=(1, 2, 3, 4))
        dz = torch.mean((flow_field[:, :, 1:, :, :] - flow_field[:, :, :-1, :, :]) ** 2, dim=(1, 2, 3, 4))
        smoothness_cost = (dx + dy + dz)
        
        magnitude_cost = torch.mean(flow_field ** 2, dim=(1, 2, 3, 4))
        
        # Adaptive penalty scaling
        # If flow is tiny (<0.5), reduce penalties to encourage movement
        # If flow is huge (>10), increase penalties to prevent overshooting
        penalty_scale = torch.where(
            flow_magnitude < 0.5,
            torch.tensor(0.1, device=self.device),  # Light penalties
            torch.where(
                flow_magnitude > 10.0,
                torch.tensor(3.0, device=self.device),  # Heavy penalties
                torch.tensor(1.0, device=self.device)   # Normal penalties
            )
        )
        
        # TOTAL REWARD
        total_reward = similarity_reward - \
                       (smoothness_weight * smoothness_cost * penalty_scale) - \
                       (magnitude_weight * magnitude_cost * penalty_scale)
        
        # For logging
        similarity_cost = 1.0 - feature_ncc
        
        # Debug logging (occasionally)
        if torch.rand(1).item() < 0.02:  # 2% of time
            grad_ncc_val = grad_ncc.mean().item()
            grad_base_val = grad_ncc_baseline.mean().item() if baseline_features is not None else 0.3
            
            feat_ncc_val = feature_ncc.mean().item()
            feat_base_val = feature_ncc_baseline.mean().item() if baseline_features is not None else 0.65
            
            grad_reward_val = gradient_reward.mean().item()
            feat_reward_val = feature_reward.mean().item()
            total_reward_val = total_reward.mean().item()
            flow_mag_val = flow_magnitude.mean().item()
            
            print(f"\n  [Reward Breakdown]")
            print(f"    Grad NCC: {grad_ncc_val:.3f} (baseline: {grad_base_val:.3f})")
            print(f"    Feat NCC: {feat_ncc_val:.3f} (baseline: {feat_base_val:.3f})")
            print(f"    Gradient reward: {grad_reward_val:.1f}")
            print(f"    Feature reward: {feat_reward_val:.1f}")
            print(f"    Total reward: {total_reward_val:.1f}")
            print(f"    Flow magnitude: {flow_mag_val:.2f}")
        
        return total_reward, similarity_cost, smoothness_cost, magnitude_cost
    
class CurriculumRewardSystem:
    """
    KEY INSIGHT: Your agent can't learn on perfect features.
    Solution: Train on progressively better features (curriculum learning)
    
    Phase 1 (Epochs 0-50): Use LOW-QUALITY features (no PCA, raw DINO)
      - These have room for improvement
      - Agent learns basic alignment
      
    Phase 2 (Epochs 50-100): Use MEDIUM-QUALITY features (high-dim PCA)
      - Agent learns finer alignment
      
    Phase 3 (Epochs 100+): Use HIGH-QUALITY features (24D PCA like DINO-Reg)
      - Agent fine-tunes on production features
    """
    def __init__(self, device):
        self.device = device
        
    def compute_ssd_loss(self, img1, img2):
        """
        Simple SSD - more sensitive than NCC for feature-based registration
        """
        return torch.mean((img1 - img2) ** 2, dim=(1, 2, 3, 4))
    
    def get_reward(self, warped_features, fixed_features, flow_field, 
                   smoothness_weight, magnitude_weight, similarity_weight,
                   baseline_features=None, epoch=0, max_epochs=150):
        """
        Use SSD instead of NCC - it's more sensitive to small changes
        """
        
        # Current alignment (SSD - lower is better)
        ssd_current = self.compute_ssd_loss(warped_features.unsqueeze(0), 
                                            fixed_features.unsqueeze(0))
        
        if baseline_features is not None:
            ssd_baseline = self.compute_ssd_loss(baseline_features.unsqueeze(0), 
                                                fixed_features.unsqueeze(0))
            # Improvement = reduction in SSD (positive is good)
            ssd_improvement = ssd_baseline - ssd_current
        else:
            # No baseline: encourage low SSD
            ssd_improvement = -ssd_current
        
        # Scale to reasonable range
        # Typical SSD ranges from 0.01 to 1.0 for features
        similarity_reward = ssd_improvement * 100.0
        
        # Penalties
        flow_magnitude = torch.mean(torch.norm(flow_field, p=2, dim=1), dim=(1, 2, 3))
        
        dx = torch.mean((flow_field[:, :, :, :, 1:] - flow_field[:, :, :, :, :-1]) ** 2, dim=(1, 2, 3, 4))
        dy = torch.mean((flow_field[:, :, :, 1:, :] - flow_field[:, :, :, :-1, :]) ** 2, dim=(1, 2, 3, 4))
        dz = torch.mean((flow_field[:, :, 1:, :, :] - flow_field[:, :, :-1, :, :]) ** 2, dim=(1, 2, 3, 4))
        smoothness_cost = (dx + dy + dz)
        
        magnitude_cost = flow_magnitude ** 2
        
        # Total reward
        total_reward = similarity_reward - \
                       (smoothness_weight * smoothness_cost) - \
                       (magnitude_weight * magnitude_cost)
        
        # Debug
        if torch.rand(1).item() < 0.02:
            print(f"\n  [SSD Reward]")
            print(f"    SSD current: {ssd_current.mean():.4f}")
            print(f"    SSD baseline: {ssd_baseline.mean():.4f if baseline_features is not None else 0:.4f}")
            print(f"    SSD improvement: {ssd_improvement.mean():.4f}")
            print(f"    Similarity reward: {similarity_reward.mean():.1f}")
            print(f"    Total reward: {total_reward.mean():.1f}")
        
        # Dummy similarity cost for logging
        similarity_cost = ssd_current
        
        return total_reward, similarity_cost, smoothness_cost, magnitude_cost
