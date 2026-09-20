import torch
import torch.nn.functional as F
import numpy as np
import math
from turtle import done
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HierarchicalRewardSystem:
    def __init__(self, device, coarse_weight_start=1.0, coarse_weight_end=0.1, 
                 annealing_epochs=25):
        self.device = device
        self.coarse_weight_start = coarse_weight_start
        self.coarse_weight_end = coarse_weight_end
        self.annealing_epochs = annealing_epochs
        
    def compute_cls_alignment_reward(self, f_mr_3d, f_ct_3d):
        """
        Compute coarse alignment reward using CLS-like global features
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
        
        # We now have a clean, direct reward signal. Scale it and apply penalties.
        # Let's use a more conservative scaling factor now that the signal is better.
        total_reward = 200 * alignment_reward - (smooth_weight * smooth_val) - (mag_weight * mag_val)
        
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
            smoothness_cost = (dz**2).mean().item() + (dy**2).mean().item() + (dx**2).mean().item()
            
            magnitude_cost = (flow_field**2).mean().item()
            
            # 2. Calculate the total weighted cost
            total_cost = (similarity_weight * similarity_cost) + \
                        (smoothness_weight * smoothness_cost) + \
                        (magnitude_weight * magnitude_cost)
                        
            # 3. The reward is the negative of the total cost. This is a stable signal.
            total_reward = -total_cost
            
            # For logging purposes, we can still return the individual components
            # Note: The similarity_reward is now also a cost (positive value)
            similarity_reward_for_logging = -similarity_cost

            return total_reward, similarity_reward_for_logging, smoothness_cost, magnitude_cost