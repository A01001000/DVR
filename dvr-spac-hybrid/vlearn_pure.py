import math
from turtle import done
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import collections
import os
import psutil
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim.lr_scheduler as lr_scheduler
import torch._dynamo

# Define a writable cache directory path
compile_cache_dir = "/disk/scratch/s2749465/torch_compile_cache"
# Create the directory if it doesn't exist
os.makedirs(compile_cache_dir, exist_ok=True)
# Set the environment variable for torch.compile to use this path
os.environ['TORCH_COMPILE_CACHE_DIR'] = compile_cache_dir

from tqdm import tqdm
from torchvision.transforms.functional import resize
import torchvision
import csv
import time
import random
from torch.amp import GradScaler
import collections
import random
import gc
import objgraph
import tracemalloc
tracemalloc.start()
from collections import deque

from dvr2.test import validate_memory_efficient
from dvr2.models_rob import VLearnPolicy_Medium, SpatialTransformer, AdaptiveFlowPolicy_Medium, Planner, Actor, VCritic, VLearnPolicy, VLearnCritic
from dvr2.utils import MINDLoss, get_warped_grid, save_slice_as_png, inspect_actor_layers, apply_flow_scaling, get_effective_spacing
from dvr2.extract_dino import extract_single_dino_features, save_feature_slice_as_png
from dvr2.reward import HierarchicalRewardSystem, RewardNormalizer, compute_smoothness_penalty, normalize_advantages, multi_scale_reward, DualChannelRewardSystem, SimplifiedSSDRewardSystem, MindSSDRewardSystem, EnhancedStableReward, ImprovementBasedReward, HybridRegistrationReward, GradientFieldReward, RawImageMINDSystem

class ReplayBuffer:
    """
    A robust, memory-efficient replay buffer using collections.deque.
    *** MODIFIED TO STORE behavior_prob FOR WIS LOSS ***
    """
    def __init__(self, capacity):
        # --- THE FIX: Use a simple Python list instead of a deque ---
        self.buffer = []
        self.capacity = capacity
        self.position = 0
        self.min_val = -5.0
        self.max_val = 5.0

    def push(self, state, action_mean, action_std, reward_item, next_state, done, sampled_action, behavior_prob):
        # (self)  (1)       (2)         (3)         (4)          (5)       (6)       (7)             (8)  <--- ADDED
        def to_uint8_numpy(tensor):
            # Detach, move to CPU, and explicitly delete the original tensor reference
            with torch.no_grad():  # Ensure no gradients
                np_array = tensor.detach().cpu().numpy()
            # Clip to the defined range
            np_array = np.clip(np_array, self.min_val, self.max_val)
            # Scale to [0, 1]
            np_array = (np_array - self.min_val) / (self.max_val - self.min_val)
            # Scale to [0, 255] and convert to uint8
            return (np_array * 255).astype(np.uint8)

        # Store old experience to explicitly delete it
        old_experience = None
        if len(self.buffer) >= self.capacity:
            old_experience = self.buffer[self.position]
            
        experience = (
            to_uint8_numpy(state),
            to_uint8_numpy(action_mean),
            to_uint8_numpy(action_std),
            float(reward_item),  
            to_uint8_numpy(next_state),
            float(done),  
            to_uint8_numpy(sampled_action),
            to_uint8_numpy(behavior_prob) # <--- ADDED
        )
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # Replace the old experience
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        
        # Explicitly delete the old experience
        if old_experience is not None:
            del old_experience
            
    def push_preformatted(self, experience_tuple):
        """
        Pushes an experience tuple that is already in the correct uint8 format.
        Assumes tuple format: (state, mean, std, reward, next_state, done, action, behavior_prob)
        """
        # Store old experience to explicitly delete it
        old_experience = None
        if len(self.buffer) >= self.capacity:
            old_experience = self.buffer[self.position]
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # Replace the old experience
        self.buffer[self.position] = experience_tuple
        self.position = (self.position + 1) % self.capacity
        
        # Explicitly delete the old experience
        if old_experience is not None:
            del old_experience
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}")
            
        experiences = random.sample(self.buffer, batch_size)
        # --- MODIFIED ZIP ---
        state, action_mean, action_std, reward, next_state, done, sampled_action, behavior_prob = zip(*experiences)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        def from_uint8_numpy(np_array_batch):
            # Stack the uint8 arrays and convert to float32
            float_batch = np.stack(np_array_batch).astype(np.float32) / 255.0
            # Scale back to the original range
            dequantized = float_batch * (self.max_val - self.min_val) + self.min_val
            # Convert to a tensor on the correct device
            return torch.from_numpy(dequantized).to(device, dtype=torch.float32, non_blocking=True)
        
        # Convert the NumPy arrays back to Torch tensors on the GPU
        state_tensor = from_uint8_numpy(state)
        next_state_tensor = from_uint8_numpy(next_state)
        mean_tensor = from_uint8_numpy(action_mean)
        std_tensor = from_uint8_numpy(action_std)
        action_tensor = from_uint8_numpy(sampled_action)
        behavior_prob_tensor = from_uint8_numpy(behavior_prob) # <--- ADDED

        if mean_tensor.dim() == 6:
            mean_tensor = mean_tensor.squeeze(1)
            std_tensor = std_tensor.squeeze(1)
            action_tensor = action_tensor.squeeze(1)
            behavior_prob_tensor = behavior_prob_tensor.squeeze(1) # <--- ADDED
            
        # Create tensors first, then move to device with non_blocking
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device, non_blocking=True)
        done_tensor = torch.tensor(done, dtype=torch.float32).to(device, non_blocking=True)
        
        # --- MODIFIED CLEANUP ---
        del experiences, state, action_mean, action_std, reward, next_state, done, sampled_action, behavior_prob
        
        # --- MODIFIED RETURN ---
        return (state_tensor, mean_tensor, std_tensor, reward_tensor, next_state_tensor, done_tensor, action_tensor, behavior_prob_tensor)

    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Explicitly clear the buffer and free memory"""
        self.buffer.clear()
        self.position = 0

def check_and_fix_model_state(model, model_name="model"):
    """Check for NaN parameters in model and reset them if found"""
    nan_found = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"WARNING: NaN found in {model_name} parameter {name}! Resetting to Xavier normal.")
            if len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            else:
                nn.init.constant_(param, 0)
            nan_found = True
    
    if nan_found:
        print(f"Model {model_name} parameters have been reset due to NaN values.")
    
    return nan_found

def create_registration_state(f_mr_warped, f_ct_fixed, flow_accumulated, flow_scale=5.0):
    """
    Creates a robustly normalized state by scaling each component independently.
    """
    with torch.no_grad():
        # --- ROBUST NORMALIZATION ---
        # Normalize each component to have zero mean and unit variance independently.
        f_mr_norm = (f_mr_warped - f_mr_warped.mean()) / (f_mr_warped.std() + 1e-8)
        f_ct_norm = (f_ct_fixed - f_ct_fixed.mean()) / (f_ct_fixed.std() + 1e-8)

        # --- THIS IS THE FIX ---
        # Your flow is tiny (~0.5), so dividing by 5.0 makes it ~0.1.
        # This makes V(s_t) and V(s_t+1) look identical to the critic.
        # We MUST normalize the flow to have a strong signal, just like the features.
        flow_norm = (flow_accumulated - flow_accumulated.mean()) / (flow_accumulated.std() + 1e-8)
        # --- END OF FIX ---

        # Concatenate the consistently scaled components along channel dimension
        state = torch.cat([f_mr_norm, f_ct_norm, flow_norm], dim=0)

        return state

import inspect

def find_leaking_tensors(epoch):
    """
    An advanced debugging function to find tensors with attached computation
    graphs that are not being released from memory.
    """
    print("\n" + "="*50)
    print(f"🔬 ADVANCED MEMORY LEAK ANALYSIS (End of Epoch {epoch+1})")
    print("="*50)

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    leaking_tensors = [
        obj for obj in gc.get_objects()
        if isinstance(obj, torch.Tensor) and obj.grad_fn is not None
    ]

    if not leaking_tensors:
        print("✅ No leaking tensors with grad_fn found.")
        return

    print(f"💥 FOUND {len(leaking_tensors)} POTENTIAL LEAKING TENSORS WITH A COMPUTATION GRAPH!")
    
    leaking_tensors.sort(key=lambda t: t.numel(), reverse=True)
    for i, tensor in enumerate(leaking_tensors[:5]):
        print(f"\n--- Leak #{i+1} ---")
        print(f"  - Tensor Shape: {tensor.shape}")
        
        # Find what is holding a reference to this tensor
        referrers = gc.get_referrers(tensor)
        print(f"  - Held by {len(referrers)} object(s):")
        for ref in referrers:
            # --- NEW: Print memory ID and more context ---
            print(f"    - Type: {type(ref).__name__}, Memory ID: {id(ref)}")
            if isinstance(ref, (list, tuple)):
                print(f"      Context (length): {len(ref)}")
            # This is an advanced check to see if the referrer is a local variable
            # in the frame of this function, which can help identify the debugger itself as the source.
            if 'find_leaking_tensors' in [f.function for f in inspect.stack() if f.frame.f_locals.get('ref') is ref]:
                 print("      💡 HINT: This referrer might be a variable within the find_leaking_tensors function itself.")

    print("="*50)
    # --- NEW: Clear the list at the end ---
    del leaking_tensors
    gc.collect()
    
def to_uint8_numpy(tensor):
    with torch.no_grad():
        np_array = tensor.detach().cpu().numpy()
    min_val, max_val = -5.0, 5.0
    np_array = np.clip(np_array, min_val, max_val)
    np_array = (np_array - min_val) / (max_val - min_val)
    
    # --- THE FINAL FIX ---
    # Force NumPy to create a new array in memory, severing all ties
    # to the original tensor's memory block.
    return (np_array * 255).astype(np.uint8).copy()

def check_gradients(model, name, max_norm=10.0):
    """Monitor and clip gradients"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    if total_norm > max_norm:
        print(f"WARNING: {name} gradient norm {total_norm:.2f} exceeds {max_norm}")
    
    return total_norm

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """Custom LR scheduler with warmup and cosine decay"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return epoch / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return max(min_lr / optimizer.defaults['lr'], cosine_decay)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def classical_refinement(flow_init, f_mr, f_ct, transformer, num_iters=100, lr=0.5, smooth_weight=5.0):
    """
    Classical gradient descent refinement after RL initialization.
    Similar to DINO-Reg's ConvexAdam but simpler.
    
    Args:
        flow_init: Initial flow from RL agent [1, 3, D, H, W]
        f_mr, f_ct: Features [1, C, D, H, W]
        transformer: SpatialTransformer
        num_iters: Number of refinement iterations
        lr: Learning rate
        smooth_weight: Smoothness weight
    
    Returns:
        Refined flow field
    """
    device = flow_init.device
    
    # Make flow optimizable
    flow = flow_init.clone().detach().requires_grad_(True)
    
    # Simple Adam optimizer
    optimizer = torch.optim.Adam([flow], lr=lr)
    
    for iter_idx in range(num_iters):
        optimizer.zero_grad()
        
        # Warp moving features
        warped_mr = transformer(f_mr, flow)
        
        # SSD loss (like DINO-Reg)
        ssd_loss = torch.mean((warped_mr - f_ct) ** 2)
        
        # Smoothness penalty (L2 like DINO-Reg)
        dx = torch.mean((flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]) ** 2)
        dy = torch.mean((flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]) ** 2)
        dz = torch.mean((flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]) ** 2)
        smooth_loss = (dx + dy + dz) * smooth_weight
        
        # Total loss
        total_loss = ssd_loss + smooth_loss
        
        # Optimize
        total_loss.backward()
        optimizer.step()
        
        # Optional: Print progress
        if iter_idx % 20 == 0:
            print(f"  Refinement iter {iter_idx}: SSD={ssd_loss.item():.4f}, Smooth={smooth_loss.item():.4f}")
    
    return flow.detach()

def add_initial_misalignment(f_mr, f_ct, device, max_displacement=10):
    """
    Add random initial displacement to create learning opportunity.
    
    This is CRITICAL: if images start perfectly aligned, there's nothing to learn!
    """
    B, C, D, H, W = f_mr.shape
    
    # Generate random displacement field
    init_flow = torch.randn(B, 3, D, H, W, device=device) * (max_displacement / 3.0)
    
    # Apply initial displacement
    transformer = SpatialTransformer().to(device)
    f_mr_displaced = transformer(f_mr, init_flow)
    
    return f_mr_displaced, init_flow

def verify_data_pairing(mr_batch, ct_batch, feature_loader, batch_idx):
    """
    Add this check at the start of your training loop to verify pairing.
    """
    # Save example slices to disk
    if batch_idx == 0:
        import matplotlib.pyplot as plt
        
        # FIX: Keep as tensors, move to CPU at the end
        mr_slice = mr_batch[0, 0, mr_batch.shape[2]//2].cpu()
        ct_slice = ct_batch[0, 0, ct_batch.shape[2]//2].cpu()
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(mr_slice.numpy(), cmap='gray')  # Convert here
        axes[0].set_title('MR Image')
        axes[1].imshow(ct_slice.numpy(), cmap='gray')  # Convert here
        axes[1].set_title('CT Image')
        plt.savefig('data_pairing_check.png')
        plt.close()
        
        print(f"\n🔍 Data pairing check saved to 'data_pairing_check.png'")
        print(f"   MR stats: min={mr_batch.min():.3f}, max={mr_batch.max():.3f}, mean={mr_batch.mean():.3f}")
        print(f"   CT stats: min={ct_batch.min():.3f}, max={ct_batch.max():.3f}, mean={ct_batch.mean():.3f}")
        
        # FIX: Use tensors for correlation
        correlation = torch.corrcoef(torch.stack([mr_slice.flatten(), ct_slice.flatten()]))[0, 1]
        print(f"   Correlation: {correlation.item():.3f}")
        if correlation > 0.9:
            print(f"   ⚠️ WARNING: Images are highly correlated! Might be comparing same modality!")

def pretrain_with_supervised_flows(planner, actor, device, num_samples=100):
    """
    Pretrain the actor to produce reasonable flows using synthetic data.
    This prevents the "do nothing" collapse.
    """
    print("\n🎓 Pretraining actor with supervised flows...")
    
    optimizer = torch.optim.Adam(
        list(planner.parameters()) + list(actor.parameters()), 
        lr=1e-4
    )
    
    for i in tqdm(range(num_samples), desc="Pretraining"):
        # Create synthetic state (random features + no flow)
        state = torch.randn(1, 5, 8, 8, 8, device=device)  # Adjust channels
        
        # Target: produce flow of magnitude ~3 voxels
        target_flow_magnitude = 3.0
        
        # Get current flow
        plan_dist = planner(state)
        plan = plan_dist.mean
        pred_flow = actor(plan)
        
        # Loss: encourage non-zero flows
        magnitude = torch.norm(pred_flow, p=2, dim=1).mean()
        loss = (magnitude - target_flow_magnitude) ** 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"✅ Pretraining complete. Final flow magnitude: {magnitude.item():.2f}")
    
def vlearn_critic_update(critic, target_critic, states, actions, rewards, 
                         next_states, dones, policy, behavior_probs, 
                         gamma=0.99, rho_clip=1.0):
    """
    CORRECT V-Learn critic update using WIS loss (Equation 4 from paper).
    *** WITH NUMERICAL STABILITY FIX ***
    """
    with torch.no_grad():
        next_action_dist = policy(next_states)
        next_actions = next_action_dist.sample()
        
        next_values = target_critic.get_value(next_states)
        targets = rewards + gamma * next_values * (1 - dones)
    
    current_values_v1, current_values_v2 = critic(states)
    
    # --- START STABILITY FIX ---
    # 1. Get log-prob sum for current policy
    current_action_log_probs = policy(states).log_prob(actions).sum(dim=[1,2,3,4], keepdim=False)
    
    # 2. Get log-prob sum for behavior policy (from buffer)
    # behavior_probs is [0,1], so log is safe.
    behavior_action_log_probs = torch.log(behavior_probs + 1e-10).sum(dim=[1,2,3,4], keepdim=False)

    # 3. Calculate log-ratio
    log_ratio = current_action_log_probs - behavior_action_log_probs.detach()

    # 4. Clamp in LOG-SPACE to avoid Inf.
    # log(rho_clip) is the max allowed log_ratio. (e.g., log(1.0) = 0.0)
    log_ratio_clipped = torch.clamp(log_ratio, max=math.log(rho_clip))
    
    # 5. Now exp() is safe and will never be > rho_clip (e.g., > 1.0)
    importance_weights = torch.exp(log_ratio_clipped)
    # --- END STABILITY FIX ---
    
    td_errors_v1 = (current_values_v1 - targets) ** 2
    td_errors_v2 = (current_values_v2 - targets) ** 2
    
    td_errors = torch.min(td_errors_v1, td_errors_v2) 
    
    # This multiplication is now safe
    weighted_loss = importance_weights * td_errors.squeeze(-1)
    critic_loss = weighted_loss.mean()
    
    return critic_loss

def vlearn_policy_update(policy, critic, states, actions, behavior_probs,
                        old_policy_params, advantages): # Pass advantages in
    """
    CORRECT V-Learn policy update.
    *** WITH NUMERICAL STABILITY FIX ***
    """
    
    action_dist = policy(states)
    
    # --- START STABILITY FIX ---
    # 1. Get log-prob sum for current policy
    current_action_log_probs = action_dist.log_prob(actions).sum(dim=[1,2,3,4], keepdim=False)
    
    # 2. Get log-prob sum for behavior policy (from buffer)
    behavior_action_log_probs = torch.log(behavior_probs + 1e-10).sum(dim=[1,2,3,4], keepdim=False)

    # 3. Calculate log-ratio
    log_ratio = current_action_log_probs - behavior_action_log_probs.detach()

    # 4. Clamp in LOG-SPACE. Policy clip is 1.0. log(1.0) = 0.0
    log_ratio_clipped = torch.clamp(log_ratio, max=0.0) # max=math.log(1.0)
    
    # 5. Now exp() is safe
    importance_ratios = torch.exp(log_ratio_clipped)
    # --- END STABILITY FIX ---
    
    # This multiplication is now safe
    policy_loss = -(importance_ratios * advantages.detach().squeeze(-1)).mean()
    
    kl_divergence = compute_kl_divergence(action_dist, old_policy_params)
    trust_region_penalty = 10.0 * F.relu(kl_divergence - 0.1)
    
    total_loss = policy_loss + trust_region_penalty
    
    return total_loss, policy_loss.item(), trust_region_penalty.item()

def compute_registration_reward(mr, ct, flow):
    """Simple NCC-based reward"""
    # Apply flow to MR
    transformer = SpatialTransformer()
    warped_mr = transformer(mr, flow)
    
    # Compute NCC
    ncc = compute_ncc(warped_mr, ct)
    
    # Simple reward
    return ncc * 100.0


def compute_kl_divergence(dist1, old_params):
    """Simplified KL computation for trust region"""
    # Implement proper KL divergence between Gaussian distributions
    return torch.tensor(0.0)  # Placeholder


def compute_ncc(img1, img2):
    """Normalized cross-correlation"""
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    img1_norm = (img1_flat - img1_flat.mean()) / (img1_flat.std() + 1e-8)
    img2_norm = (img2_flat - img2_flat.mean()) / (img2_flat.std() + 1e-8)
    
    return torch.mean(img1_norm * img2_norm)

def run_registration_episode_hybrid(planner, actor, critic, target_critic, transformer, f_mr_3d, f_ct_3d,
                                    replay_buffer, optimizers, scaler, 
                                    epoch, epochs, current_max_steps, total_steps, hyperparams, 
                                    current_spacing=None, use_refinement=True):
    """
    HYBRID APPROACH: RL for coarse initialization + Classical refinement
    
    1. RL produces a COARSE flow (fewer steps, larger actions)
    2. Classical optimization refines it (like DINO-Reg)
    3. Reward is based on coarse alignment quality, not final registration
    """
    episode_metrics = {
        'total_reward': 0, 'reward_count': 0, 'sim_cost': 0,
        'smooth_cost': 0, 'mag_cost': 0, 'total_loss': 0,
        'planner_loss': 0, 'critic_loss': 0, 'alpha_loss': 0, 
        'decoder_loss': 0, 'update_count': 0,
        'refinement_improvement': 0  
    }
    
    n_step = 3
    n_step_buffer = deque(maxlen=n_step)
    
    FEATURE_SHAPE_LOW = hyperparams['FEATURE_SHAPE_LOW']
    C_feat = 64
    
    flow_acc = torch.zeros((1, 3, *hyperparams['vol_shape']), device=f_mr_3d.device)
    
    f_mr_low = F.interpolate(f_mr_3d.unsqueeze(0), size=FEATURE_SHAPE_LOW, mode='trilinear', align_corners=False).squeeze(0)
    f_ct_low = F.interpolate(f_ct_3d.unsqueeze(0), size=FEATURE_SHAPE_LOW, mode='trilinear', align_corners=False).squeeze(0)

    # CRITICAL: Store baseline features (unwarped) for improvement calculation
    baseline_features_full = f_mr_3d.unsqueeze(0).detach()
    
    # RL PHASE: Coarse alignment (fewer steps than before)
    # CRITICAL: Use fewer RL steps since classical refinement will finish the job
    rl_steps = min(current_max_steps, 3)

    for step in range(current_max_steps):
        
        # 1. CREATE STATE
        with torch.no_grad():
            flow_for_state = F.interpolate(flow_acc, size=FEATURE_SHAPE_LOW, mode='trilinear', align_corners=False)
            flow_for_state = apply_flow_scaling(flow_for_state, flow_acc.shape[2:], FEATURE_SHAPE_LOW, current_spacing)
            
            warped_f_mr_low = transformer(f_mr_low.unsqueeze(0), flow_for_state).squeeze(0)
            
            current_state_low = create_registration_state(
                f_mr_warped=warped_f_mr_low,
                f_ct_fixed=f_ct_low, 
                flow_accumulated=flow_for_state.squeeze(0)
            ).unsqueeze(0)

        # 2. GET PLAN & ACTION
        plan_dist = planner(current_state_low)
        current_plan = plan_dist.sample()
        
        incremental_flow_low = actor(current_plan)
        
        # 3. UPSAMPLE & APPLY ACTION
        incremental_flow_full = F.interpolate(
            incremental_flow_low,
            size=flow_acc.shape[2:],
            mode='trilinear',
            align_corners=False
        )
        incremental_flow_full = apply_flow_scaling(incremental_flow_full, incremental_flow_low.shape[2:], flow_acc.shape[2:])

        new_flow_acc = flow_acc + incremental_flow_full

        # 4. GET REWARD (WITH BASELINE FOR IMPROVEMENT)
        with torch.no_grad():
            warped_feat_full = transformer(f_mr_3d.unsqueeze(0), new_flow_acc).squeeze(0)
            
            # CRITICAL: Pass baseline features to reward system
            reward_tensor, sim_cost_tensor, smooth_cost_tensor, mag_cost_tensor = hyperparams["reward_system"].get_reward(
                warped_features=warped_feat_full.unsqueeze(0), 
                fixed_features=f_ct_3d.unsqueeze(0), 
                flow_field=new_flow_acc, 
                smoothness_weight=hyperparams["SMOOTHNESS_WEIGHT"], 
                magnitude_weight=hyperparams["MAGNITUDE_WEIGHT"], 
                similarity_weight=hyperparams["SIMILARITY_WEIGHT"],
                baseline_features=baseline_features_full,  
                epoch=epoch,
                max_epochs=epochs
            )

            reward = reward_tensor.item()
            sim_cost = sim_cost_tensor.item()
            smooth_cost = smooth_cost_tensor.item()
            mag_cost = mag_cost_tensor.item()
            
            episode_metrics['total_reward'] += reward
            episode_metrics['reward_count'] += 1
            episode_metrics['sim_cost'] += sim_cost
            episode_metrics['smooth_cost'] += smooth_cost
            episode_metrics['mag_cost'] += mag_cost
                    
        done = (step == rl_steps - 1)
        
        # 5. GET NEXT STATE
        with torch.no_grad():
            flow_for_next_state = F.interpolate(new_flow_acc, size=FEATURE_SHAPE_LOW, mode='trilinear', align_corners=False)
            flow_for_next_state = apply_flow_scaling(flow_for_next_state, new_flow_acc.shape[2:], FEATURE_SHAPE_LOW, current_spacing)
            
            next_warped_f_mr_low = transformer(f_mr_low.unsqueeze(0), flow_for_next_state).squeeze(0)

            next_state_low = create_registration_state(
                f_mr_warped=next_warped_f_mr_low,
                f_ct_fixed=f_ct_low,
                flow_accumulated=flow_for_next_state.squeeze(0)
            ).unsqueeze(0)

        # 6. STORE EXPERIENCE
        experience = (
            to_uint8_numpy(current_state_low.squeeze(0).clone().detach().cpu()),
            to_uint8_numpy(plan_dist.mean.clone().detach().cpu()),
            to_uint8_numpy(plan_dist.stddev.clone().detach().cpu()),
            reward, 
            to_uint8_numpy(next_state_low.squeeze(0).clone().detach().cpu()),
            float(done),
            to_uint8_numpy(current_plan.clone().detach().cpu())
        )
        n_step_buffer.append(experience)
        
        if len(n_step_buffer) == n_step:
            rewards = [transition[3] for transition in n_step_buffer]
            n_step_reward = sum(hyperparams['GAMMA'] ** i * r for i, r in enumerate(rewards))
                
            first_state, first_mean, first_std, _, _, _, first_plan = n_step_buffer[0]
            final_next_state = n_step_buffer[-1][4]
            is_terminal = n_step_buffer[-1][5]

            experience = (
                first_state, first_mean, first_std,
                n_step_reward,
                final_next_state,
                is_terminal,
                first_plan
            )
            replay_buffer.push_preformatted(experience)
            del experience, first_state, first_mean, first_std, first_plan, final_next_state
                    
        flow_acc = new_flow_acc.detach()
        total_steps += 1
                    
        del current_state_low, next_state_low, plan_dist, current_plan, incremental_flow_low
        del incremental_flow_full, flow_for_state, warped_f_mr_low, reward, sim_cost, smooth_cost, mag_cost
        
        
        # 7. OPTIMIZATION STEP (same as before but with better scaling)
        if len(replay_buffer) >= hyperparams['BATCH_SIZE'] and total_steps % hyperparams['UPDATE_EVERY_N_STEPS'] == 0:
            
            s, old_p_mean, old_p_std, r, s_prime, d, p = replay_buffer.sample(hyperparams['BATCH_SIZE'])
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                
                # CRITICAL: NO REWARD SCALING - rewards are already in correct range
                with torch.no_grad():
                    next_value = target_critic.get_value(s_prime)
                    value_target = r.view(-1, 1) + hyperparams["GAMMA"] * next_value * (1 - d.view(-1, 1))

                current_value1, current_value2 = critic(s)
                
                # WIS calculation
                old_dist_plan = torch.distributions.Normal(old_p_mean, old_p_std)
                new_dist_plan_critic = planner(s)
                new_log_p = new_dist_plan_critic.log_prob(p).sum(dim=-1, keepdim=True)
                old_log_p = old_dist_plan.log_prob(p).sum(dim=-1, keepdim=True)
                log_rho = (new_log_p - old_log_p.detach())
                rho = torch.exp(torch.clamp(log_rho, math.log(0.1), math.log(5.0)))
                rho = rho / (rho.mean().detach() + 1e-8)
                
                # Critic loss
                critic1_loss = F.smooth_l1_loss(current_value1, value_target.detach(), reduction='none')
                critic2_loss = F.smooth_l1_loss(current_value2, value_target.detach(), reduction='none')
                critic_loss = (critic1_loss * rho.detach() + critic2_loss * rho.detach()).mean()
                
                # Advantage calculation
                v_current = torch.min(current_value1, current_value2)
                advantage = value_target - v_current.detach()
                advantage_normalized = advantage / (advantage.abs().mean() + 1e-8)
                advantage_normalized = torch.clamp(advantage_normalized, -3.0, 3.0)
                
                # Planner loss
                new_dist_plan_actor = planner(s)
                plan_for_loss = new_dist_plan_actor.rsample()
                new_log_p_actor = new_dist_plan_actor.log_prob(plan_for_loss).sum(dim=-1, keepdim=True)
                
                entropy = new_dist_plan_actor.entropy().sum(dim=-1).mean()
                alpha_loss = -(hyperparams["log_alpha"] * (entropy.detach() + hyperparams["target_entropy"])).mean()
                
                planner_loss = -(new_log_p_actor * advantage_normalized.detach()).mean() - \
                               (hyperparams["log_alpha"].exp().detach() * entropy)
                
                # Decoder loss (unchanged)
                generated_flow_low = actor(plan_for_loss)
                generated_flow_for_loss = F.interpolate(generated_flow_low, size=hyperparams['vol_shape'], mode='trilinear', align_corners=False)
                generated_flow_for_loss = apply_flow_scaling(generated_flow_for_loss, generated_flow_low.shape[2:], hyperparams['vol_shape'])

                dx = torch.mean((generated_flow_for_loss[:, :, :, :, 1:] - generated_flow_for_loss[:, :, :, :, :-1]) ** 2)
                dy = torch.mean((generated_flow_for_loss[:, :, :, 1:, :] - generated_flow_for_loss[:, :, :, :-1, :]) ** 2)
                dz = torch.mean((generated_flow_for_loss[:, :, 1:, :, :] - generated_flow_for_loss[:, :, :-1, :, :]) ** 2)
                smoothness_penalty = (dx + dy + dz)
                magnitude_penalty = torch.mean(generated_flow_for_loss ** 2)

                decoder_loss = smoothness_penalty + 0.1 * magnitude_penalty

                total_loss = critic_loss + planner_loss + alpha_loss + decoder_loss
        
                episode_metrics['total_loss'] += total_loss.item()
                episode_metrics['planner_loss'] += planner_loss.item()
                episode_metrics['critic_loss'] += critic_loss.item()
                episode_metrics['alpha_loss'] += alpha_loss.item()
                episode_metrics['decoder_loss'] += decoder_loss.item()
                episode_metrics['update_count'] += 1

                if not torch.isfinite(total_loss):
                    print(f"WARNING: Invalid loss detected! Skipping update.")
                    continue
            
            # Backpropagation
            total_loss_scaled = total_loss / hyperparams["GRADIENT_ACCUMULATION_STEPS"]
            scaler.scale(total_loss_scaled).backward()

            if (total_steps % hyperparams["GRADIENT_ACCUMULATION_STEPS"]) == 0:
                scaler.unscale_(optimizers["critic"])
                scaler.unscale_(optimizers["planner"])
                scaler.unscale_(optimizers["actor"])
                scaler.unscale_(optimizers["alpha"])
                
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(planner.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)

                scaler.step(optimizers["critic"])
                scaler.step(optimizers["alpha"])
                scaler.step(optimizers["planner"])
                scaler.step(optimizers["actor"])
                
                scaler.update()
                
                optimizers["critic"].zero_grad()
                optimizers["alpha"].zero_grad()
                optimizers["planner"].zero_grad()
                optimizers["actor"].zero_grad()
                        
                # Update target network
                with torch.no_grad():
                    for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                        target_param.data.mul_(hyperparams["POLYAK_TAU"])
                        target_param.data.add_((1 - hyperparams["POLYAK_TAU"]) * param.data)
                        
                del s, old_p_mean, old_p_std, r, s_prime, d, p, value_target, current_value1, current_value2
                del rho, critic_loss, advantage, new_log_p_actor, entropy, alpha_loss, planner_loss
                del generated_flow_low, smoothness_penalty, decoder_loss, total_loss, total_loss_scaled
                  
        if done:
            break
    
    # CLASSICAL REFINEMENT PHASE 
    if use_refinement and epoch > 10:  # Only refine after initial learning
        print(f"  Running classical refinement...")
        
        # Measure quality before refinement
        with torch.no_grad():
            warped_before = transformer(f_mr_3d.unsqueeze(0), flow_acc)
            ncc_before = torch.mean((warped_before - f_ct_3d.unsqueeze(0)) ** 2).item()
        
        # Refine using classical optimization
        flow_refined = classical_refinement(
            flow_init=flow_acc,
            f_mr=f_mr_3d.unsqueeze(0),
            f_ct=f_ct_3d.unsqueeze(0),
            transformer=transformer,
            num_iters=100,  # Like DINO-Reg but fewer iterations
            lr=0.5,
            smooth_weight=5.0  # Like DINO-Reg
        )
        
        # Measure improvement
        with torch.no_grad():
            warped_after = transformer(f_mr_3d.unsqueeze(0), flow_refined)
            ncc_after = torch.mean((warped_after - f_ct_3d.unsqueeze(0)) ** 2).item()
            
        improvement = ncc_before - ncc_after
        episode_metrics['refinement_improvement'] = improvement
        print(f"  Refinement improved SSD by: {improvement:.4f}")
        
        flow_acc = flow_refined
        
    # Handle remaining n-step buffer
    if len(n_step_buffer) > 0:
        final_next_state_numpy = n_step_buffer[-1][4]
        min_val, max_val = -5.0, 5.0
        final_next_state_float = final_next_state_numpy.astype(np.float32) / 255.0
        dequantized = final_next_state_float * (max_val - min_val) + min_val
        
        state_tensor = torch.from_numpy(dequantized).unsqueeze(0).to(hyperparams["device"])
        final_v_next = target_critic.get_value(state_tensor)
        G = final_v_next.detach()
    else:
        G = torch.tensor(0.0, device=hyperparams["device"])
    
    while len(n_step_buffer) > 0:
        for transition in reversed(list(n_step_buffer)):
            reward = transition[3]
            G = reward + hyperparams['GAMMA'] * G

        first_state, first_mean, first_std, _, _, is_terminal, first_action = n_step_buffer[0]
        n_step_reward = G.item()
        final_next_state = n_step_buffer[-1][4]
        is_terminal = n_step_buffer[-1][5]

        experience = (
            first_state, first_mean, first_std,
            n_step_reward,
            final_next_state,
            is_terminal,
            first_action
        )
        replay_buffer.push_preformatted(experience)
        n_step_buffer.popleft()
        
        del first_state, first_mean, first_std, is_terminal, first_action, n_step_reward, final_next_state, experience

    del n_step_buffer, f_mr_low, f_ct_low, flow_acc, G, baseline_features_full

    return new_flow_acc.detach(), total_steps, episode_metrics

def train_vlearn(dino_encoder, pca_transformer, feature_loader, val_loader, device, epochs=100, max_steps=5, save_dir="checkpoints", dino_dir=None, is_continue_training=False, use_raw_images=False, dataset_type="abdomen"):
    """
    Train V-Learn policy using the correct WIS-loss implementation,
    while leveraging the data loading, logging, and validation from vlearn_rob.py
    """
    torch.autograd.set_detect_anomaly(True)
    
    # --- V-Learn Paper Hyperparameters (from vlearn_rob.py) ---
    IMG_SIZE = 128
    GAMMA = 0.99
    POLYAK_TAU = 0.005 # Target network update rate (from proper file)
    vol_shape = [128, 128, 128]
    C_feat = 64
    FEATURE_SHAPE_LOW = (8, 8, 8) 
    
    if use_raw_images:
        STATE_CHANNELS = 5 # 1 (MR) + 1 (CT) + 3 (Flow)
        pca_dims = 1 
        print(f"Running with raw images. Setting model input channels to: {STATE_CHANNELS} (1+1+3)")
    else:
        if hasattr(pca_transformer, 'n_components_'):
            pca_dims = pca_transformer.n_components_
        else:
            pca_dims = 24
        STATE_CHANNELS = (pca_dims * 2) + 3
        print(f"Running with DINO features. Setting model input channels to: {STATE_CHANNELS} ({pca_dims}x2 + 3 flow)")
    
    BATCH_SIZE = 32
    BUFFER_CAPACITY = 10000
    UPDATE_EVERY_N_STEPS = 1
    GRADIENT_ACCUMULATION_STEPS = 1
    POLICY_UPDATE_INTERVAL = 2 # Delayed policy updates
    
    LR_POLICY = 1e-5
    LR_CRITIC = 3e-5
    
    gradient_reward_system = GradientFieldReward(device)
    raw_image_mind_reward_system = RawImageMINDSystem(device)
    
    reward_system = raw_image_mind_reward_system if use_raw_images else gradient_reward_system
    
    hyperparams_reward = {
        'SMOOTHNESS_WEIGHT': 0.1,
        'MAGNITUDE_WEIGHT': 0.05,
        'SIMILARITY_WEIGHT': 1.0,
    }

    # --- Initialize Models (from proper_vlearn_implementation.py) ---
    policy = VLearnPolicy(state_dim=STATE_CHANNELS, action_dim=3).to(device)
    critic = VLearnCritic(state_dim=STATE_CHANNELS).to(device)
    target_critic = VLearnCritic(state_dim=STATE_CHANNELS).to(device)
    target_critic.load_state_dict(critic.state_dict())
    
    transformer = SpatialTransformer().to(device)
    
    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad) + sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"📊 Parameters (V-Learn): {total_params:,}")

    # --- Initialize Optimizers ---
    policy_optimizer = Adam(policy.parameters(), lr=LR_POLICY)
    critic_optimizer = Adam(critic.parameters(), lr=LR_CRITIC)

    # Schedulers
    WARMUP_EPOCHS = 15
    policy_scheduler = get_cosine_schedule_with_warmup(
        policy_optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=epochs, min_lr=1e-6
    )
    critic_scheduler = get_cosine_schedule_with_warmup(
        critic_optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=epochs, min_lr=1e-6
    )
    policy_warmup_scheduler = lr_scheduler.LinearLR(policy_optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    critic_warmup_scheduler = lr_scheduler.LinearLR(critic_optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    policy_main_scheduler = lr_scheduler.SequentialLR(policy_optimizer, schedulers=[policy_warmup_scheduler, policy_scheduler], milestones=[WARMUP_EPOCHS])
    critic_main_scheduler = lr_scheduler.SequentialLR(critic_optimizer, schedulers=[critic_warmup_scheduler, critic_scheduler], milestones=[WARMUP_EPOCHS])

    
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
    scaler = GradScaler()
    
    torch.cuda.empty_cache()
    
    total_steps = 0
    best_val_mind = float('inf')
    patience = 40  
    patience_counter = 0

    checkpoint_path = os.path.join(save_dir, "best_vlearn_model.pth")
    log_path = os.path.join(save_dir, "training_log.csv")
    start_epoch = 0
    
    # --- Checkpoint Loading ---
    if os.path.exists(checkpoint_path) and is_continue_training:
        print(f"✅ Resuming training by loading weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        policy.load_state_dict(checkpoint['policy_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        
        policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] 
        best_val_mind = checkpoint.get('best_val_mind', float('inf'))

        print(f"✅ Resumed successfully. Starting at Epoch {start_epoch + 1}.")
    else:
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Avg_Critic_Loss", "Avg_Policy_Loss", "Avg_Reward", "Avg_Flow_Magnitude", "Val_MIND", "Total_Updates_This_Epoch",
                             "Avg_Sim_Cost", "Avg_Smooth_Penalty", "Avg_Magnitude_Penalty", 
                             "Critic Learning Rate", "Policy Learning Rate"])
        print("ℹ️ No checkpoint found or needed. Starting training from scratch.")
        
    print("🚀 Starting V-Learn training with PROPER WIS-LOSS implementation...")
    
    # --- Compile Models ---
    print("🚀 Compiling model for a significant speed boost...")
    policy = torch.compile(policy)
    critic = torch.compile(critic)
    target_critic = torch.compile(target_critic)
    transformer = torch.compile(transformer)
    
    snapshot1 = None # For memory tracing
    
    # --- Main Training Loop ---
    for epoch in range(start_epoch, epochs):
        
        epoch_start_time = time.time()
        epoch_total_critic_loss = 0
        epoch_total_policy_loss = 0
        epoch_total_reward = 0
        epoch_total_flow_magnitude = 0
        epoch_update_count = 0
        epoch_reward_count = 0
        epoch_sample_count = 0
        epoch_sim_cost, epoch_smooth_penalty, epoch_mag_penalty = 0, 0, 0
                
        print(f"\n🎯 Epoch {epoch+1}/{epochs}")
        
        # Curriculum learning
        if epoch < 30:
            current_max_steps = 3
        elif epoch < 80:
            current_max_steps = 5
        else:
            current_max_steps = max_steps
            
        # Use full resolution
        current_resolution = (128, 128, 128)
        vol_shape = current_resolution
        D_vol, H_vol, W_vol = vol_shape

        # --- Data Loading Loop ---
        for batch_idx, (mr_batch, ct_batch, batch_spacing) in enumerate(tqdm(feature_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            mr_batch, ct_batch = mr_batch.to(device), ct_batch.to(device)
            batch_size = mr_batch.size(0)
            
            if epoch == 0:
                verify_data_pairing(mr_batch, ct_batch, feature_loader, batch_idx)
            
            if current_resolution != (128, 128, 128):
                mr_batch = F.interpolate(mr_batch, size=current_resolution, mode='trilinear')
                ct_batch = F.interpolate(ct_batch, size=current_resolution, mode='trilinear')
                batch_spacing = [get_effective_spacing(sp, (128,128,128), current_resolution) for sp in batch_spacing]

            # --- Feature Extraction ---
            if not use_raw_images:
                with torch.no_grad():
                    f_mr_batch = []
                    f_ct_batch = []
                    for i in range(batch_size):
                        mr_feat, ct_feat = extract_single_dino_features(dino_encoder, mr_batch.squeeze(1)[i], ct_batch.squeeze(1)[i], pca_transformer)
                        f_mr_batch.append(mr_feat.detach())
                        f_ct_batch.append(ct_feat.detach())
                        del mr_feat, ct_feat
                    f_mr_batch = torch.stack(f_mr_batch, dim=0).to(device)
                    f_ct_batch = torch.stack(f_ct_batch, dim=0).to(device)
            else:
                f_mr_batch = mr_batch
                f_ct_batch = ct_batch
            
            # --- Initial Misalignment ---
            if epoch < 30:
                f_mr_batch, init_flow = add_initial_misalignment(
                    f_mr_batch, f_ct_batch, device, max_displacement=10
                )
            
            # --- Episode Loop ---
            for i in range(batch_size):
                with torch.no_grad():
                    f_mr_3d = f_mr_batch[i] # This is [C_feat, D_feat, H_feat, W_feat]
                    f_ct_3d = f_ct_batch[i]
                    current_spacing = batch_spacing[i]
                
                # Low-res features for state
                f_mr_low = F.interpolate(f_mr_3d.unsqueeze(0), size=FEATURE_SHAPE_LOW, mode='trilinear', align_corners=False).squeeze(0)
                f_ct_low = F.interpolate(f_ct_3d.unsqueeze(0), size=FEATURE_SHAPE_LOW, mode='trilinear', align_corners=False).squeeze(0)
                
                flow_acc = torch.zeros((1, 3, D_vol, H_vol, W_vol), device=device)
                
                # --- Run V-Learn Episode ---
                for step in range(current_max_steps):
                    
                    # 1. CREATE STATE
                    with torch.no_grad():
                        flow_for_state = F.interpolate(flow_acc, size=FEATURE_SHAPE_LOW, mode='trilinear', align_corners=False)
                        flow_for_state = apply_flow_scaling(flow_for_state, flow_acc.shape[2:], FEATURE_SHAPE_LOW, current_spacing)
                        warped_f_mr_low = transformer(f_mr_low.unsqueeze(0), flow_for_state).squeeze(0)
                        
                        current_state_low = create_registration_state(
                            f_mr_warped=warped_f_mr_low,
                            f_ct_fixed=f_ct_low, 
                            flow_accumulated=flow_for_state.squeeze(0)
                        ).unsqueeze(0)

                    # 2. GET ACTION
                    action_dist = policy(current_state_low)
                    action_low = action_dist.sample()
                    
                    with torch.no_grad():
                        # Store exp(log_prob) for WIS
                        behavior_prob = action_dist.log_prob(action_low).exp().detach()

                    # 3. APPLY ACTION
                    incremental_flow_full = F.interpolate(
                        action_low,
                        size=flow_acc.shape[2:],
                        mode='trilinear',
                        align_corners=False
                    )
                    incremental_flow_full = apply_flow_scaling(incremental_flow_full, action_low.shape[2:], flow_acc.shape[2:])
                    new_flow_acc = flow_acc + incremental_flow_full

                    # 4. GET REWARD
                    with torch.no_grad():
                        # Reward is based on full-res features/images
                        warped_feat_full = transformer(f_mr_3d.unsqueeze(0), new_flow_acc).squeeze(0)
                        reward_tensor, sim_cost_tensor, smooth_cost_tensor, mag_cost_tensor = reward_system.get_reward(
                            warped_features=warped_feat_full.unsqueeze(0), 
                            fixed_features=f_ct_3d.unsqueeze(0), 
                            flow_field=new_flow_acc, 
                            smoothness_weight=hyperparams_reward["SMOOTHNESS_WEIGHT"], 
                            magnitude_weight=hyperparams_reward["MAGNITUDE_WEIGHT"], 
                            similarity_weight=hyperparams_reward["SIMILARITY_WEIGHT"],
                            baseline_features=f_mr_3d.unsqueeze(0),
                            epoch=epoch,
                            max_epochs=epochs
                        )
                        reward = reward_tensor.item()
                        
                        epoch_total_reward += reward
                        epoch_reward_count += 1
                        epoch_sim_cost += sim_cost_tensor.item()
                        epoch_smooth_penalty += smooth_cost_tensor.item()
                        epoch_mag_penalty += mag_cost_tensor.item()
                                
                    done = (step == current_max_steps - 1)
                    
                    # 5. GET NEXT STATE
                    with torch.no_grad():
                        flow_for_next_state = F.interpolate(new_flow_acc, size=FEATURE_SHAPE_LOW, mode='trilinear', align_corners=False)
                        flow_for_next_state = apply_flow_scaling(flow_for_next_state, new_flow_acc.shape[2:], FEATURE_SHAPE_LOW, current_spacing)
                        next_warped_f_mr_low = transformer(f_mr_low.unsqueeze(0), flow_for_next_state).squeeze(0)
                        next_state_low = create_registration_state(
                            f_mr_warped=next_warped_f_mr_low,
                            f_ct_fixed=f_ct_low,
                            flow_accumulated=flow_for_next_state.squeeze(0)
                        ).unsqueeze(0)

                    # 6. STORE EXPERIENCE
                    replay_buffer.push(
                        current_state_low.squeeze(0),
                        action_dist.mean.squeeze(0),
                        action_dist.stddev.squeeze(0),
                        reward,
                        next_state_low.squeeze(0),
                        float(done),
                        action_low.squeeze(0),
                        behavior_prob.squeeze(0)
                    )
                            
                    flow_acc = new_flow_acc.detach()
                    total_steps += 1
                    
                    del current_state_low, next_state_low, action_dist, action_low, behavior_prob
                    del incremental_flow_full, flow_for_state, warped_f_mr_low
                    
                    # 7. OPTIMIZATION STEP
                    if len(replay_buffer) >= BATCH_SIZE and total_steps % UPDATE_EVERY_N_STEPS == 0:
                        
                        states, means, stds, rewards, next_states, dones, actions, behavior_probs = replay_buffer.sample(BATCH_SIZE)
                        
                        rewards = rewards.unsqueeze(-1)
                        dones = dones.unsqueeze(-1)
                        
                        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                            # === CRITIC UPDATE ===
                            critic_loss = vlearn_critic_update(
                                critic, target_critic, states, actions, rewards,
                                next_states, dones, policy, behavior_probs, GAMMA
                            )
                        
                        critic_loss_scaled = critic_loss / GRADIENT_ACCUMULATION_STEPS
                        scaler.scale(critic_loss_scaled).backward()
                        epoch_total_critic_loss += critic_loss.item()
                        
                        # === POLICY UPDATE (Delayed) ===
                        if total_steps % POLICY_UPDATE_INTERVAL == 0:
                            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                                # Re-compute advantage: r + gamma*V(s') - V(s)
                                with torch.no_grad():
                                    next_values = target_critic.get_value(next_states)
                                    targets = rewards + GAMMA * next_values * (1 - dones)
                                    current_values = critic.get_value(states)
                                    advantages = targets - current_values
                                
                                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                                
                                old_params = {k: v.clone() for k, v in policy.state_dict().items()}
                                
                                policy_loss, _, _ = vlearn_policy_update(
                                    policy, critic, states, actions, behavior_probs, old_params, advantages
                                )
                            
                            policy_loss_scaled = policy_loss / GRADIENT_ACCUMULATION_STEPS
                            scaler.scale(policy_loss_scaled).backward()
                            
                            epoch_total_policy_loss += policy_loss.item()
                        
                        epoch_update_count += 1
                        
                        # --- GRADIENT ACCUMULATION STEP ---
                        if total_steps % GRADIENT_ACCUMULATION_STEPS == 0:
                            scaler.unscale_(critic_optimizer)
                            scaler.unscale_(policy_optimizer)
                            
                            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

                            scaler.step(critic_optimizer)
                            if total_steps % POLICY_UPDATE_INTERVAL == 0:
                                scaler.step(policy_optimizer)
                            
                            scaler.update()
                            
                            critic_optimizer.zero_grad()
                            policy_optimizer.zero_grad()
                                    
                            # === TARGET NETWORK UPDATE (POLYAK) ===
                            with torch.no_grad():
                                for param, target_param in zip(critic.parameters(), 
                                                               target_critic.parameters()):
                                    target_param.data.mul_(1 - POLYAK_TAU)
                                    target_param.data.add_(POLYAK_TAU * param.data)
                        
                        del states, means, stds, rewards, next_states, dones, actions, behavior_probs
                        
                    if done:
                        break
                # --- End of Episode ---
                
                per_voxel_magnitudes = torch.norm(flow_acc.detach(), p=2, dim=1)
                epoch_total_flow_magnitude += per_voxel_magnitudes.mean().item()
                epoch_sample_count += 1
                
                del f_mr_3d, f_ct_3d, f_mr_low, f_ct_low, flow_acc
            
            del mr_batch, ct_batch, f_mr_batch, f_ct_batch
            torch.cuda.empty_cache()
            gc.collect()
                
        # --- End of Epoch Processing ---
        avg_critic_loss = epoch_total_critic_loss / epoch_update_count if epoch_update_count > 0 else 0
        avg_policy_loss = epoch_total_policy_loss / (epoch_update_count / POLICY_UPDATE_INTERVAL) if epoch_update_count > 0 else 0
        avg_reward = epoch_total_reward / epoch_reward_count if epoch_reward_count > 0 else 0
        avg_flow_magnitude = epoch_total_flow_magnitude / epoch_sample_count if epoch_sample_count > 0 else 0
        avg_sim_cost = epoch_sim_cost / epoch_reward_count if epoch_reward_count > 0 else 0
        avg_smooth_penalty = epoch_smooth_penalty / epoch_reward_count if epoch_reward_count > 0 else 0
        avg_mag_penalty = epoch_mag_penalty / epoch_reward_count if epoch_reward_count > 0 else 0

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        check_and_fix_model_state(policy, "policy")
        critic_nan = check_and_fix_model_state(critic, "critic")
        if critic_nan:
            target_critic.load_state_dict(critic.state_dict())
        
        # --- Validation ---
        print(f"\n📊 Running memory-efficient validation for epoch {epoch+1}...")
        torch.cuda.empty_cache()
        gc.collect()
        
        max_val_samples = 5 if epoch < epochs - 10 else 10
        
        # --- THIS IS THE FIX ---
        # Removed the wrapper, calling the updated function directly
        # with the correct arguments.
        current_val_mind = validate_memory_efficient(
            policy=policy,
            critic=critic,
            val_loader=val_loader, 
            device=device,
            vol_shape=vol_shape,
            feature_shape=(C_feat, *FEATURE_SHAPE_LOW), # Pass low-res shape
            max_steps=current_max_steps,
            max_validation_samples=max_val_samples,
            use_raw_images=use_raw_images,
            dino_encoder=dino_encoder,
            pca_transformer=pca_transformer,
            dataset_type=dataset_type
        )
        # --- END OF FIX ---
        
        policy_main_scheduler.step()
        critic_main_scheduler.step()
        current_lr_policy = policy_optimizer.param_groups[0]['lr']
        current_lr_critic = critic_optimizer.param_groups[0]['lr']

        torch.cuda.empty_cache()
        gc.collect() 

        # --- Logging ---
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_critic_loss, avg_policy_loss, avg_reward, avg_flow_magnitude, current_val_mind, epoch_update_count,
                             avg_sim_cost, avg_smooth_penalty, avg_mag_penalty, 
                             current_lr_critic, current_lr_policy])

        print(f"\nEpoch {epoch+1}/{epochs} Summary ({epoch_duration/60:.2f} min):")
        print(f"  LRs -> Critic: {current_lr_critic:.2e}, Policy: {current_lr_policy:.2e}")
        print(f"  Avg Critic Loss: {avg_critic_loss:.4f}")
        print(f"  Avg Policy Loss: {avg_policy_loss:.4f}")
        print(f"  Avg Reward: {avg_reward:.4f}")
        print(f"  Avg Flow Magnitude: {avg_flow_magnitude:.4f}")
        print(f"  Val MIND: {current_val_mind:.4f}")
        print(f"  Updates: {epoch_update_count}")
        print(f"  Avg Smooth Penalty: {avg_smooth_penalty:.4f}")
        print(f"  Avg Mag Penalty: {avg_mag_penalty:.4f}")
        
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"  🧠 Memory Usage: RSS={(mem_info.rss / (1024 ** 3)):.2f} GB | VMS={(mem_info.vms / (1024 ** 3)):.2f} GB")

        # --- Checkpoint Saving ---
        if current_val_mind < best_val_mind:
            best_val_mind = current_val_mind
            print(f"   💾 New best model saved! Val MIND: {best_val_mind:.4f}")
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'best_val_mind': best_val_mind,
                'policy_state_dict': policy.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'target_critic_state_dict': target_critic.state_dict(),
                'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(save_dir, "best_vlearn_model.pth"))
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement. Patience: {patience_counter}/{patience}")
        
        if (epoch + 1) % 25 == 0:
            checkpoint_path_e = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'best_val_mind': best_val_mind,
                'policy_state_dict': policy.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'target_critic_state_dict': target_critic.state_dict(),
                'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
            }, checkpoint_path_e)
            print(f"   💾 Checkpoint saved: {checkpoint_path_e}")
            target_critic.load_state_dict(critic.state_dict())
        
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {patience} epochs without improvement.")
            break
          
        # --- Memory Tracing ---
        print("\n" + "="*50)
        print(f"🔬 TRACEMALLOC ANALYSIS (End of Epoch {epoch+1})")
        print("="*50)
        snapshot2 = tracemalloc.take_snapshot()
        if snapshot1 is not None:
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            print("Top 10 memory differences since last epoch (potential leaks):")
            for stat in top_stats[:10]:
                print(stat)
        snapshot1 = snapshot2
        
    # --- Final Save ---
    torch.save({
                'epoch': epoch + 1,
                'best_val_mind': best_val_mind,
                'policy_state_dict': policy.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'target_critic_state_dict': target_critic.state_dict(),
                'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
            }, os.path.join(save_dir, "final_vlearn_model.pth"))
    print(f"\n🏁 Training completed!")
    print(f"   Best validation MIND: {best_val_mind:.4f}")
    print(f"   Models saved in: {save_dir}")
    
    return policy