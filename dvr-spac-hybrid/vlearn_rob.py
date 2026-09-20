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
from dvr2.models_rob import VLearnPolicy_Medium, SpatialTransformer, AdaptiveFlowPolicy_Medium, Planner, Actor, VCritic
from dvr2.utils import MINDLoss, get_warped_grid, save_slice_as_png, inspect_actor_layers, apply_flow_scaling, get_effective_spacing
from dvr2.extract_dino import extract_single_dino_features, save_feature_slice_as_png
from dvr2.reward import HierarchicalRewardSystem, RewardNormalizer, compute_smoothness_penalty, normalize_advantages, multi_scale_reward, DualChannelRewardSystem, SimplifiedSSDRewardSystem, MindSSDRewardSystem, EnhancedStableReward, ImprovementBasedReward, HybridRegistrationReward, GradientFieldReward, RawImageMINDSystem

class ReplayBuffer:
    """
    A robust, memory-efficient replay buffer using collections.deque.
    """
    def __init__(self, capacity):
        # --- THE FIX: Use a simple Python list instead of a deque ---
        self.buffer = []
        self.capacity = capacity
        self.position = 0
        self.min_val = -5.0
        self.max_val = 5.0

    def push(self, state, action_mean, action_std, reward_item, next_state, done, sampled_action):
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
            to_uint8_numpy(sampled_action)
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
        """Pushes an experience tuple that is already in the correct uint8 format."""
        self.buffer.append(experience_tuple)
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}")
            
        experiences = random.sample(self.buffer, batch_size)
        state, action_mean, action_std, reward, next_state, done, sampled_action = zip(*experiences)
        
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

        if mean_tensor.dim() == 6:
            mean_tensor = mean_tensor.squeeze(1)
            std_tensor = std_tensor.squeeze(1)
            action_tensor = action_tensor.squeeze(1)
            
        # Create tensors first, then move to device with non_blocking
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device, non_blocking=True)
        done_tensor = torch.tensor(done, dtype=torch.float32).to(device, non_blocking=True)
        
        # Clean up intermediate variables
        del experiences, state, action_mean, action_std, reward, next_state, done, sampled_action
        
        return (state_tensor, mean_tensor, std_tensor, reward_tensor, next_state_tensor, done_tensor, action_tensor)

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
    
    Key insight: Importance weight goes OUTSIDE the squared error,
    not inside the Bellman target!
    """
    with torch.no_grad():
        # Sample actions from current policy for next state
        next_action_dist = policy(next_states)
        next_actions = next_action_dist.sample()
        
        # V-function target (no importance weighting here!)
        next_values = target_critic.get_value(next_states)
        targets = rewards + gamma * next_values * (1 - dones)
    
    # Current V-function prediction
    current_values = critic.get_value(states)
    
    # Compute importance weights
    current_action_probs = policy(states).log_prob(actions).exp()
    importance_weights = (current_action_probs / (behavior_probs + 1e-8)).detach()
    
    # Clip importance weights (paper uses ε_ρ = 1.0)
    importance_weights = torch.clamp(importance_weights, 0, rho_clip)
    
    # === KEY: WIS LOSS (Equation 4) ===
    # Importance weight OUTSIDE the squared error!
    td_errors = (current_values - targets) ** 2
    weighted_loss = importance_weights * td_errors
    critic_loss = weighted_loss.mean()
    
    return critic_loss

def vlearn_policy_update(policy, critic, states, actions, behavior_probs,
                        old_policy_params, epsilon_mean=0.1, epsilon_cov=0.0005):
    """
    CORRECT V-Learn policy update with TRPL trust region.
    
    Uses advantage from V-function (not Q-function).
    """
    # Compute advantages using V-function
    with torch.no_grad():
        # We need next states for bootstrap, but for simplicity
        # we'll use 1-step advantage here
        current_values = critic.get_value(states)
        # In practice, you'd use: r + γV(s') - V(s)
        # For now, simplified version:
        advantages = -current_values  # Placeholder, needs proper computation
    
    # Normalize advantages (paper recommendation)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Compute policy distribution
    action_dist = policy(states)
    
    # Importance sampling for off-policy correction
    current_log_probs = action_dist.log_prob(actions)
    importance_ratios = (current_log_probs.exp() / (behavior_probs + 1e-8))
    
    # Clip importance ratios
    importance_ratios = torch.clamp(importance_ratios, 0, 1.0)
    
    # Policy gradient loss with importance sampling
    policy_loss = -(importance_ratios * advantages).mean()
    
    # TRPL trust region constraint (see paper Section 3.3)
    # This is simplified - use actual TRPL layer from Otto et al. 2021
    kl_divergence = compute_kl_divergence(action_dist, old_policy_params)
    trust_region_penalty = 10.0 * F.relu(kl_divergence - epsilon_mean)
    
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
    Train V-Learn policy with configurable network architectures
    """
    torch.autograd.set_detect_anomaly(True)
    
    # --- V-Learn Paper Hyperparameters with Memory-Safe Options ---
    IMG_SIZE = 128
    GAMMA = 0.99  # Discount factor for future rewards
    POLYAK_TAU = 0.01 # Target network update rate
    vol_shape = [128, 128, 128]
    feature_shape = [64, 128, 16, 16]
    C_feat = 64
    LATENT_DIM = 32
    FEATURE_SHAPE_LOW = (8, 8, 8) # The low-res shape your models work at
    pca_dims = 24  
    FEATURE_CHANNELS = 8
    
    if use_raw_images:
        # Use raw images: 1 channel (MR) + 1 channel (CT) + 3 channels (Flow)
        STATE_CHANNELS = 5
        pca_dims = 1 # Placeholder, not used by models
        print(f"Running with raw images. Setting model input channels to: {STATE_CHANNELS} (1+1+3)")
    else:
        # Use DINO features: (pca_dims * 2) + 3 channels
        if hasattr(pca_transformer, 'n_components_'):
            pca_dims = pca_transformer.n_components_
            print(f"Detected PCA dimensions from transformer: {pca_dims}")
        else:
            pca_dims = 24
            print(f"Could not detect PCA dimensions, defaulting to: {pca_dims}")

        if pca_dims != 24:
            print(f"WARNING: You wanted 24 dims, but loaded transformer has {pca_dims}!")
            print("    If this is unintended, delete 'pca_transformer.pkl' and restart.")
            
        STATE_CHANNELS = (pca_dims * 2) + 3
        print(f"Running with DINO features. Setting model input channels to: {STATE_CHANNELS} ({pca_dims}x2 + 3 flow)")
    
    BATCH_SIZE = 32 # Memory-optimized buffer capacity based on network size
    BUFFER_CAPACITY = 10000  # V-Learn paper standard for large setups
    UPDATE_EVERY_N_STEPS = 1  # More frequent updates to prevent staleness
    
    # COMPENSATION STRATEGIES for reduced batch sizes:
    # 1. Accumulate gradients to simulate larger effective batch size
    # GRADIENT_ACCUMULATION_STEPS = max(1, 64 // BATCH_SIZE)  # Aim for effective batch ~64
    GRADIENT_ACCUMULATION_STEPS = 1
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    
    # 2. Scale learning rate based on EFFECTIVE batch size (not physical batch size)
    
    MAX_FLOW_MAGNITUDE = 10.0
    # LEARNING ENCOURAGEMENT: New weight for progress-based reward
    IMPROVEMENT_WEIGHT = 50.0 
    SIMILARITY_WEIGHT = 1.0
    NGF_WEIGHT = 0.15
    BETA = 0.9
    
    # PENALTY SCHEDULE
    PENALTY_WARMUP_EPOCHS = 15      # Use low penalties for the first 15 epochs
    PENALTY_ANNEALING_EPOCHS = 75   # Gradually increase penalties until epoch 75
    SMOOTHNESS_WEIGHT_START = 0.5   # The low value that worked well initially
    SMOOTHNESS_WEIGHT_END = 5.0     # The high value to enforce refinement later
    MAGNITUDE_WEIGHT_START = 0.2    # The low value that worked well initially
    MAGNITUDE_WEIGHT_END = 20.0     # The high value to enforce refinement later
    
    # Learning Rate Schedule
    WARMUP_EPOCHS = 15  # Number of epochs for the high-exploration phase

    # Actor learning rate schedule
    LR_ACTOR_START = 5e-5
    LR_ACTOR_MAX = 1e-4
    LR_ACTOR_END = 5e-5
    
    # Critic learning rate schedule
    LR_CRITIC_START = 5e-5
    LR_CRITIC_MAX = 1e-4
    LR_CRITIC_END = 5e-5

    # You can also schedule the entropy bonus (alpha)
    LOG_ALPHA_START = 0.0   # Corresponds to alpha = 1.0
    LOG_ALPHA_END = -5.0    # Corresponds to a small alpha, encouraging exploitation
    ANNEALING_EPOCHS = 75 # Number of epochs to transition from exploration to exploitation
    SIMILARITY_PHASE_EPOCHS = 75 # Number of epochs to focus only on alignment
    
    # Cap for flows
    STRICT_NORM_LIMIT = 250.0
    RELAXED_NORM_LIMIT = 500.0
    
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    
    # Initialize models
    planner = Planner(in_channels=STATE_CHANNELS, latent_dim=LATENT_DIM).to(device)
    actor = Actor(latent_dim=LATENT_DIM, feature_channels=FEATURE_CHANNELS).to(device)
    critic = VCritic(in_channels=STATE_CHANNELS).to(device)
    
    # V-Learn Target Critic
    target_critic = VCritic(in_channels=STATE_CHANNELS).to(device)
    
    # Initialize to output near-zero
    for param in critic.parameters():
        if len(param.shape) > 1:
            nn.init.orthogonal_(param, gain=0.01)
        else:
            nn.init.constant_(param, 0)
    target_critic.load_state_dict(critic.state_dict())
    
    # Initialize critic to output values near 0
    for param in critic.parameters():
        if len(param.shape) > 1:
            nn.init.orthogonal_(param, gain=0.01)
        else:
            nn.init.constant_(param, 0)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()
    
    transformer = SpatialTransformer().to(device)
    
    hierarchical_reward_system = HierarchicalRewardSystem(device)
    dual_reward_system = DualChannelRewardSystem(device)
    simplified_reward_system = SimplifiedSSDRewardSystem(device)
    mind_reward_system = MindSSDRewardSystem(device)
    enhanced_reward_system = EnhancedStableReward(device)
    improvement_reward_system = ImprovementBasedReward(device)
    hybrid_reward_system = HybridRegistrationReward(device)
    gradient_reward_system = GradientFieldReward(device)
    raw_image_mind_reward_system = RawImageMINDSystem(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in planner.parameters() if p.requires_grad) + sum(p.numel() for p in actor.parameters() if p.requires_grad) + sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"📊 Parameters: {total_params:,}")
    
    LR_PLANNER = 1e-5  
    LR_ACTOR = 1e-5    
    LR_CRITIC = 3e-5  
    LR_ALPHA = 1e-6   

    planner_optimizer = Adam(planner.parameters(), lr=LR_PLANNER)
    actor_optimizer = Adam(actor.parameters(), lr=LR_ACTOR)
    critic_optimizer = Adam(critic.parameters(), lr=LR_CRITIC)
    log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
    alpha_optimizer = Adam([log_alpha], lr=LR_ALPHA)
    
    # critic_scheduler = ReduceLROnPlateau(critic_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # actor_scheduler = ReduceLROnPlateau(actor_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    planner_scheduler = get_cosine_schedule_with_warmup(
        planner_optimizer, warmup_epochs=15, total_epochs=epochs, min_lr=1e-6
    )
    critic_scheduler = get_cosine_schedule_with_warmup(
        critic_optimizer, warmup_epochs=15, total_epochs=epochs, min_lr=1e-6
    )
    
    # Sequential scheduler to handle the warmup phase
    planner_warmup_scheduler = lr_scheduler.LinearLR(planner_optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    critic_warmup_scheduler = lr_scheduler.LinearLR(critic_optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)

    # Chain the warmup and cosine schedulers together
    planner_main_scheduler = lr_scheduler.SequentialLR(planner_optimizer, schedulers=[planner_warmup_scheduler, planner_scheduler], milestones=[WARMUP_EPOCHS])
    critic_main_scheduler = lr_scheduler.SequentialLR(critic_optimizer, schedulers=[critic_warmup_scheduler, critic_scheduler], milestones=[WARMUP_EPOCHS])
    
    # Adaptive Entropy (SAC-style)
    # The target entropy is a heuristic. A common choice is -|A|, the negative dimensionality of the action space.
    # Since your action space is a 3D flow field, we can approximate this.
    # Let's assume the low-res action space is 3x16x16x16 = 12288
    target_entropy = -8.0 # This is a heuristic value, can be tuned based on your action space
    # target_entropy = -np.prod((3, D_feat // 32, H_feat // 32, W_feat // 32)).item()
    
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
    # Use PyTorch 2.x GradScaler with proper syntax
    scaler = GradScaler()  # Standard initialization for PyTorch 
    
    # Memory optimization settings - conservative approach
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                
    torch.cuda.empty_cache()  # Clear cache for large networks
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  
    
    total_steps = 0
    best_loss = float('inf')
    best_val_mind = float('inf')

    patience = 40  
    patience_counter = 0
    
    # Initialize gradient accumulation tracking
    accumulated_gradients = 0

    checkpoint_path = os.path.join(save_dir, "best_vlearn_model.pth") # Let's resume from the best model
    log_path = os.path.join(save_dir, "training_log.csv")
    start_epoch = 0 # Default start epoch
    
    if os.path.exists(checkpoint_path) and is_continue_training:
        print(f"✅ Resuming training by loading weights from: {checkpoint_path}")
        
        # Load the dictionary from the file
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # --- Restore the States ---
        
        # Restore model weights
        planner.load_state_dict(checkpoint['planner_state_dict'])
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        
        # Restore optimizer states (important for momentum, etc.)
        planner_optimizer.load_state_dict(checkpoint['planner_optimizer_state_dict'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        # Restore the training progress
        start_epoch = checkpoint['epoch'] 
        best_val_mind = checkpoint.get('best_val_mind', float('inf')) # Use .get for safety

        print(f"✅ Resumed successfully. Starting at Epoch {start_epoch + 1}.")
        
    else:
        # Enhanced logging
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Avg_Update_Loss", "Avg_Reward", "Avg_Flow_Magnitude", "Val_MIND", "Total_Updates_This_Epoch", "Current_Alpha", 
                             "Avg_Total_Loss", "Avg_Planner_Loss", "Avg_Critic_Loss", "Avg_Alpha_Loss", "Avg_Decoder_Loss", "Avg_Sim_Cost", "Avg_Smooth_Penalty", 
                             "Avg_Magnitude_Penalty", "Avg_Refinement_Improvement", "Critic Learning Rate", "Planner Learning Rate"])
        print("ℹ️ No checkpoint found or needed. Starting training from scratch.")
        
    print("🚀 Starting V-Learn training with PERFORMANCE-OPTIMIZED architecture...")
    print(f"📊 Training parameters (optimized for memory + performance balance):")
    print(f"   - Network: {total_params:,} parameters")
    print(f"   - Epochs: {epochs}")
    print(f"   - Max steps per episode: {max_steps}")
    print(f"   - Batch size: {BATCH_SIZE} (V-Learn paper: 64)")
    print(f"   - Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} (via gradient accumulation)")
    print(f"   - Buffer capacity: {BUFFER_CAPACITY} (V-Learn paper: 500K)")
    print(f"   - Max flow magnitude: {MAX_FLOW_MAGNITUDE}")
    print(f"   - Architecture: V-Learn original with ReLU")
    print(f"   - Twin Critics: ✅ Enabled (reduces overestimation bias)")
    print(f"   - Advantage Normalization: ✅ Enabled")
    print(f"   - Delayed Policy Updates: ✅ Enabled (every 2 critic updates)")
    print(f"   - Gradient Accumulation: ✅ {GRADIENT_ACCUMULATION_STEPS} steps (simulates larger batch)")
    print(f"   - Memory Optimizations: ✅ CPU buffer + periodic cleanup")
    
    if GRADIENT_ACCUMULATION_STEPS > 1:
        print(f"💡 Performance Note: Gradient accumulation maintains training stability despite smaller physical batch size")
    
    # Create a variable to hold the memory snapshot from the previous epoch
    snapshot1 = None
    
    # Track flow magnitude to ensure it stays reasonable
    flow_magnitude_history = []
    
    # Normalize rewards
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
    # reward_normalizer = RewardNormalizer(num_inputs=1) 
    scaler = GradScaler()
    
    # Multiple resolutions for robustness
    resolutions = {"low": (64, 64, 64), "full": (128, 128, 128)}

    # Bundle optimizers and some hyperparams for easier passing
    optimizers = {
        'planner': planner_optimizer, 
        'actor': actor_optimizer, 
        'critic': critic_optimizer, 
        'alpha': alpha_optimizer
    }
    
    # Compile models
    print("🚀 Compiling model for a significant speed boost...")
    planner = torch.compile(planner)
    actor = torch.compile(actor)
    critic = torch.compile(critic)
    target_critic = torch.compile(target_critic)
    transformer = torch.compile(transformer)
    
    hyperparams = {
        'BATCH_SIZE': BATCH_SIZE,
        'UPDATE_EVERY_N_STEPS': UPDATE_EVERY_N_STEPS,
        'GAMMA': GAMMA,
        'POLYAK_TAU': POLYAK_TAU,
        'BETA': BETA,
        'target_entropy': target_entropy,
        'GRADIENT_ACCUMULATION_STEPS': GRADIENT_ACCUMULATION_STEPS,
        'reward_system': raw_image_mind_reward_system if use_raw_images else gradient_reward_system,
        'SIMILARITY_WEIGHT': SIMILARITY_WEIGHT,
        'SMOOTHNESS_WEIGHT': SMOOTHNESS_WEIGHT_START,
        'MAGNITUDE_WEIGHT': MAGNITUDE_WEIGHT_START,
        'device': device,
        'log_alpha': log_alpha, 
        'MAX_FLOW_MAGNITUDE': MAX_FLOW_MAGNITUDE,
        'FEATURE_SHAPE_LOW': FEATURE_SHAPE_LOW, 
        'vol_shape': vol_shape,
        'pca_dims': pca_dims
    }
    
    # --- Main Training Loop ---
    for epoch in range(epochs):
        # --- Periodically reset the buffer ---
        if (epoch + 1) % 50 == 0:
            print(f"\n🧠 Clearing replay buffer at epoch {epoch+1} to refresh experiences.")
            replay_buffer.clear()
            gc.collect()
        
        epoch_start_time = time.time()
        epoch_total_loss = 0
        epoch_total_reward = 0
        epoch_total_flow_magnitude = 0
        epoch_update_count = 0
        epoch_reward_count = 0
        epoch_sample_count = 0
        epoch_pos_ncc_reward, epoch_neg_ncc_reward, epoch_smooth_penalty, epoch_mag_penalty, epoch_improvement, epoch_ngf_penalty = 0, 0, 0, 0, 0, 0
        epoch_actor_loss, epoch_critic_loss, epoch_alpha_loss = 0, 0, 0
        epoch_total_sim_reward, epoch_total_coarse_sim, epoch_total_fine_sim = 0, 0, 0
        epoch_sim_cost, epoch_planner_loss, epoch_decoder_loss, epoch_refinement_improvement = 0, 0, 0, 0
                
        print(f"\n🎯 Epoch {epoch+1}/{epochs}")
        """
        # --- MANUAL LEARNING RATE SCHEDULING ---
        if epoch < WARMUP_EPOCHS:
            # --- WARMUP PHASE ---
            progress = epoch / WARMUP_EPOCHS
            # Linearly increase LR from START to MAX
            current_lr_actor = LR_ACTOR_START + progress * (LR_ACTOR_MAX - LR_ACTOR_START)
            current_lr_critic = LR_CRITIC_START + progress * (LR_CRITIC_MAX - LR_CRITIC_START)

        else:
            # --- DECAY PHASE ---
            # Cosine decay from MAX to END
            progress = (epoch - WARMUP_EPOCHS) / (epochs - WARMUP_EPOCHS)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            current_lr_actor = LR_ACTOR_END + (LR_ACTOR_MAX - LR_ACTOR_END) * cosine_decay
            current_lr_critic = LR_CRITIC_END + (LR_CRITIC_MAX - LR_CRITIC_END) * cosine_decay

        # Apply the calculated learning rates to the optimizers
        for param_group in actor_optimizer.param_groups:
            param_group['lr'] = current_lr_actor
        for param_group in critic_optimizer.param_groups:
            param_group['lr'] = current_lr_critic
        
        if epoch < PENALTY_WARMUP_EPOCHS:
            hyperparams["SMOOTHNESS_WEIGHT"] = SMOOTHNESS_WEIGHT_START
            hyperparams["MAGNITUDE_WEIGHT"] = MAGNITUDE_WEIGHT_START
        elif epoch < PENALTY_ANNEALING_EPOCHS:
            progress = (epoch - PENALTY_WARMUP_EPOCHS) / (PENALTY_ANNEALING_EPOCHS - PENALTY_WARMUP_EPOCHS)
            hyperparams["SMOOTHNESS_WEIGHT"] = SMOOTHNESS_WEIGHT_START + progress * (SMOOTHNESS_WEIGHT_END - SMOOTHNESS_WEIGHT_START)
            hyperparams["MAGNITUDE_WEIGHT"] = MAGNITUDE_WEIGHT_START + progress * (MAGNITUDE_WEIGHT_END - MAGNITUDE_WEIGHT_START)
        else:
            hyperparams["SMOOTHNESS_WEIGHT"] = SMOOTHNESS_WEIGHT_END
            hyperparams["MAGNITUDE_WEIGHT"] = MAGNITUDE_WEIGHT_END

        if epoch < SIMILARITY_PHASE_EPOCHS:
            # Phase 1: Teach the agent ONLY how to align images.
            # Penalties are turned off.
            hyperparams["SIMILARITY_WEIGHT"] = 50.0
            hyperparams["SMOOTHNESS_WEIGHT"] = 0.0
            hyperparams["MAGNITUDE_WEIGHT"] = 0.0
            if epoch == 0:
                print("\n--- CURRICULUM PHASE 1: SIMILARITY ONLY ---")
        else:
            # Phase 2: Now that it knows how to align, teach it to be smooth.
            # Turn on the penalties.
            hyperparams["SIMILARITY_WEIGHT"] = 50.0
            hyperparams["SMOOTHNESS_WEIGHT"] = 5.0
            hyperparams["MAGNITUDE_WEIGHT"] = 2.0
            if epoch == SIMILARITY_PHASE_EPOCHS:
                print("\n--- CURRICULUM PHASE 2: INTRODUCING PENALTIES ---")
        """

        # 4. Pretrain before main loop (one-time)
        if epoch == 0 and not is_continue_training:
            pretrain_with_supervised_flows(planner, actor, device)
        
        # 5. Much lighter penalties initially
        if epoch < 30:
            hyperparams["SMOOTHNESS_WEIGHT"] = 0.01  # Very light!
            hyperparams["MAGNITUDE_WEIGHT"] = 0.005  # Very light!
        elif epoch < 60:
            hyperparams["SMOOTHNESS_WEIGHT"] = 0.05
            hyperparams["MAGNITUDE_WEIGHT"] = 0.02
        else:
            hyperparams["SMOOTHNESS_WEIGHT"] = 0.1
            hyperparams["MAGNITUDE_WEIGHT"] = 0.05
        
        flow_magnitude = flow_magnitude_history[epoch-1] if epoch > 0 else 1.0
        # Adaptive scaling based on flow magnitude
        if flow_magnitude < 0.1:  # Too little movement
            hyperparams["SMOOTHNESS_WEIGHT"] *= 0.5
            hyperparams["MAGNITUDE_WEIGHT"] *= 0.5
        elif flow_magnitude > 10.0:  # Too much movement
            hyperparams["SMOOTHNESS_WEIGHT"] *= 2.0
            hyperparams["MAGNITUDE_WEIGHT"] *= 2.0
        
        # Curriculum learning
        if epoch < 30:
            # Phase 1: Learn coarse alignment with gradient rewards
            current_max_steps = 3
            use_refinement = False
            print(f"\n📚 Phase 1 (Epoch {epoch+1}): Learning gradient alignment")
        elif epoch < 80:
            # Phase 2: Add light refinement
            current_max_steps = 3
            use_refinement = True
            refinement_iters = 50
            print(f"\n📚 Phase 2 (Epoch {epoch+1}): RL + light refinement")
        else:
            # Phase 3: Full refinement
            current_max_steps = 3
            use_refinement = True
            refinement_iters = 100
            print(f"\n📚 Phase 3 (Epoch {epoch+1}): RL + full refinement")

        """
        # Calculate current progress (a value from 0.0 to 1.0)
        progress = min(1.0, epoch / PENALTY_ANNEALING_EPOCHS)
        
        # Linearly increase penalties over time
        current_smooth_weight = SMOOTH_WEIGHT_START + progress * (SMOOTH_WEIGHT_END - SMOOTH_WEIGHT_START)
        current_mag_weight = MAG_WEIGHT_START + progress * (MAG_WEIGHT_END - MAG_WEIGHT_START)

        # Set the hyperparams for this epoch
        hyperparams["SIMILARITY_WEIGHT"] = SIMILARITY_WEIGHT
        hyperparams["SMOOTHNESS_WEIGHT"] = current_smooth_weight
        hyperparams["MAGNITUDE_WEIGHT"] = current_mag_weight
        """
    
        if epoch < ANNEALING_EPOCHS:
            # Linearly decrease log_alpha over the annealing period
            progress = epoch / ANNEALING_EPOCHS
            current_log_alpha = LOG_ALPHA_START - progress * (LOG_ALPHA_START - LOG_ALPHA_END)
            hyperparams["log_alpha"].data.fill_(current_log_alpha)
        else:
            # Keep alpha low after annealing is done
            hyperparams["log_alpha"].data.fill_(LOG_ALPHA_END)
        
        
        phase_one_epochs = 75
        # Curriculum learning; multi-resolution
        if epoch < phase_one_epochs:
            current_resolution = resolutions["low"]
            current_max_steps = 3
            if epoch == 0: print("--- Phase 1: MAX STEPS: 3 ---")
        else:
            current_resolution = resolutions["full"]
            current_max_steps = 8
            if epoch == phase_one_epochs: print("\n--- Phase 2: MAX STEPS: 8 ---")
            
        """
        current_max_steps = 1
        current_resolution = resolutions["full"] # Use full resolution
        
        if epoch == 0:
            print(f"\n--- RUNNING AS SINGLE-STEP PREDICTION (max_steps = 1) ---")
        """
            
        vol_shape = current_resolution
        D_vol, H_vol, W_vol = vol_shape
        hyperparams['vol_shape'] = vol_shape

        # The feature_loader provides batches of pre-computed feature tensors
        for batch_idx, (mr_batch, ct_batch, batch_spacing) in enumerate(tqdm(feature_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            mr_batch, ct_batch = mr_batch.to(device), ct_batch.to(device)
            batch_size = mr_batch.size(0)
            
            # 2. Add data verification (first epoch only)
            if epoch == 0:
                verify_data_pairing(mr_batch, ct_batch, feature_loader, batch_idx)
            
            # on the fly downsampling for diff resolutions 
            if current_resolution != resolutions["full"]:
                mr_batch = F.interpolate(mr_batch, size=current_resolution, mode='trilinear')
                ct_batch = F.interpolate(ct_batch, size=current_resolution, mode='trilinear')
                
                # Update effective spacing
                batch_spacing = [get_effective_spacing(sp, resolutions["full"], current_resolution) for sp in batch_spacing]

            # Extract features ON THE FLY 
            if not use_raw_images:
                # Extract features ON THE FLY 
                with torch.no_grad():  # Ensure no gradients are computed for the extractor
                    f_mr_batch = []
                    f_ct_batch = []
                    for i in range(batch_size):
                        mr_feat, ct_feat = extract_single_dino_features(dino_encoder, mr_batch.squeeze(1)[i], ct_batch.squeeze(1)[i], pca_transformer)
                        f_mr_batch.append(mr_feat.detach())
                        f_ct_batch.append(ct_feat.detach())

                        if i == 0: # Only save one example to avoid clutter
                            # save_feature_slice_as_png(mr_feat, os.path.join(save_dir, "mr_feature_viz.png"))
                            # save_feature_slice_as_png(ct_feat, os.path.join(save_dir, "ct_feature_viz.png"))
                            feature_shape = mr_feat.shape
                            C_feat, D_feat, H_feat, W_feat = feature_shape
        
                            if D_vol != D_feat:
                                print(f"Warning: Volume depth ({D_vol}) and Feature depth ({D_feat}) are different!")
            
                        del mr_feat, ct_feat  # Free memory immediately
                    f_mr_batch = torch.stack(f_mr_batch, dim=0).to(device)
                    f_ct_batch = torch.stack(f_ct_batch, dim=0).to(device)
            else:
                # Use raw images directly as "features"
                f_mr_batch = mr_batch
                f_ct_batch = ct_batch
                # Set feature_shape based on raw image shape (C, D, H, W)
                # mr_batch shape is [B, 1, D, H, W], so [1:] is [1, D, H, W]
                feature_shape = f_mr_batch.shape[1:] 
                C_feat, D_feat, H_feat, W_feat = feature_shape
            
            # 3. Add initial misalignment (first 30 epochs)
            if epoch < 30:
                f_mr_batch, init_flow = add_initial_misalignment(
                    f_mr_batch, f_ct_batch, device, max_displacement=10
                )
                print(f"  Added initial misalignment: {init_flow.norm(p=2, dim=1).mean():.2f} voxels")
                
            # if epoch == 0 and batch_idx == 0:
            #     run_gradient_descent_test(f_mr_batch[0], f_ct_batch[0], device)

            # --- APPLY MIXUP AUGMENTATION ---
            alpha = 0.4 # Hyperparameter, typically between 0.1 and 0.4
            lam = np.random.beta(alpha, alpha) # Sample the mixing coefficient

            # Create a shuffled version of the batch to mix with
            indices = torch.randperm(batch_size, device=device)

            # Mix the feature tensors
            mixed_f_mr = lam * f_mr_batch + (1 - lam) * f_mr_batch[indices, :]
            mixed_f_ct = lam * f_ct_batch + (1 - lam) * f_ct_batch[indices, :]

            for i in range(batch_size):
                with torch.no_grad():
                    # Ensure no gradients are computed for the feature tensors
                    # Get single feature tensors from the batch
                    # Reshape features into 3D volumes
                    
                    # mr_img = mr_batch[i].unsqueeze(0) 
                    # ct_img = ct_batch[i].unsqueeze(0)
                    
                    # Get the final features for sample i
                    # f_mr_final_i = f_mr_batch[i]
                    # f_ct_final_i = f_ct_batch[i]
                    
                    # -- COMMENT PUT TO DO MIXUP AUGMENTATION --
                    # f_mr_3d = mixed_f_mr[i] # Use the mixed features for the state
                    # f_ct_3d = mixed_f_ct[i]
                    
                    f_mr_3d = f_mr_batch[i]  # No mixing
                    f_ct_3d = f_ct_batch[i]  # No mixing
                    
                    current_spacing = batch_spacing[i]
                
                flow_acc = torch.zeros((1, 3, D_vol, H_vol, W_vol), device=device)
                
                # Use hybrid approach
                use_refinement = (epoch > 10)  # Enable refinement after warmup

                # --- Registration Episode & Optimization Loop ---
                flow_acc, total_steps, episode_metrics = run_registration_episode_hybrid(
                    planner, actor, critic, target_critic, transformer, 
                    f_mr_3d, f_ct_3d, replay_buffer, optimizers, scaler,
                    epoch, epochs, max_steps, total_steps, hyperparams, 
                    current_spacing, use_refinement=use_refinement
                )
                
                # Track refinement benefit
                if episode_metrics['refinement_improvement'] > 0:
                    print(f"  Refinement benefit: {episode_metrics['refinement_improvement']:.4f}")
                    epoch_refinement_improvement += episode_metrics['refinement_improvement']

                per_voxel_magnitudes = torch.norm(flow_acc.detach(), p=2, dim=1) # Calculate magnitude along channel dim
                epoch_total_flow_magnitude += per_voxel_magnitudes.mean().item()
                epoch_sample_count += 1
                epoch_total_reward += episode_metrics['total_reward']
                epoch_reward_count += episode_metrics['reward_count']
                epoch_sim_cost += episode_metrics['sim_cost']
                epoch_smooth_penalty += episode_metrics['smooth_cost']
                epoch_mag_penalty += episode_metrics['mag_cost']
                epoch_total_loss += episode_metrics['total_loss']
                epoch_planner_loss += episode_metrics['planner_loss']
                epoch_critic_loss += episode_metrics['critic_loss']
                epoch_alpha_loss += episode_metrics['alpha_loss']
                epoch_decoder_loss += episode_metrics['decoder_loss']
                epoch_update_count += episode_metrics['update_count']  
                # epoch_total_sim_reward += episode_metrics['total_sim_reward']
                # epoch_total_coarse_sim += episode_metrics['coarse_sim']
                # epoch_total_fine_sim += episode_metrics['fine_sim']

                # Save deformation field and warped image after final step of this volume
                train_folder = os.path.join(save_dir, "train")
                os.makedirs(train_folder, exist_ok=True)
                if epoch % 10 == 0 or epoch == epochs - 1: # Example: save every 10 epochs
                    torch.save(flow_acc.detach().cpu(), os.path.join(train_folder, f"flow_epoch{epoch+1}_sample{i+1}.pt"))
                    
                current_volume_index = batch_idx * mixed_f_mr.shape[0] + i + 1
                
                # More aggressive memory cleanup to prevent OOM
                if current_volume_index % 20 == 0:  # Every 20 volumes (increased frequency)
                    # print(f"   🧹 Memory cleanup at volume {current_volume_index}")
                    torch.cuda.empty_cache()
                    gc.collect()
                # del mr_img_low_res, ct_img_low_res
                del f_mr_3d, f_ct_3d, flow_acc, episode_metrics
            del mixed_f_mr, mixed_f_ct, mr_batch, ct_batch, f_mr_batch, f_ct_batch
            torch.cuda.empty_cache()
            gc.collect()
                
        # --- End of Batch Processing ---         
        # End of batch - more aggressive cleanup
        torch.cuda.empty_cache()
        gc.collect()
                
        # --- End of Epoch Processing ---
        print(f"\n📊 Epoch {epoch+1} Stats: epoch_total_loss={epoch_total_loss:.6f}, epoch_update_count={epoch_update_count}")
        avg_loss_per_update = epoch_total_loss / epoch_update_count if epoch_update_count > 0 else 0
        avg_reward = epoch_total_reward / epoch_reward_count if epoch_reward_count > 0 else 0
        avg_flow_magnitude = epoch_total_flow_magnitude / epoch_sample_count if epoch_sample_count > 0 else 0
        avg_sim_cost = epoch_sim_cost / epoch_reward_count if epoch_reward_count > 0 else 0
        avg_smooth_penalty = epoch_smooth_penalty / epoch_reward_count if epoch_reward_count > 0 else 0
        avg_mag_penalty = epoch_mag_penalty / epoch_reward_count if epoch_reward_count > 0 else 0
        avg_total_loss = epoch_total_loss / epoch_update_count if epoch_update_count > 0 else 0
        avg_planner_loss = epoch_planner_loss / epoch_update_count if epoch_update_count > 0 else 0
        avg_critic_loss = epoch_critic_loss / epoch_update_count if epoch_update_count > 0 else 0
        avg_alpha_loss = epoch_alpha_loss / epoch_update_count if epoch_update_count > 0 else 0
        avg_decoder_loss = epoch_decoder_loss / epoch_update_count if epoch_update_count > 0 else 0
        avg_refinement_improvement = epoch_refinement_improvement / epoch_sample_count if epoch_sample_count > 0 else 0
        current_alpha = log_alpha.exp().item()
        # avg_total_sim_reward =  epoch_total_sim_reward / epoch_reward_count if epoch_reward_count > 0 else 0
        # avg_total_coarse_sim = epoch_total_coarse_sim / epoch_reward_count if epoch_reward_count > 0 else 0
        # avg_total_fine_sim = epoch_total_fine_sim / epoch_reward_count if epoch_reward_count > 0 else 0

        # Track flow magnitude history and check for runaway training
        flow_magnitude_history.append(avg_flow_magnitude)
        
        # Check if flow magnitude is increasing consistently (sign of instability)
        if len(flow_magnitude_history) >= 5:
            recent_trend = sum(flow_magnitude_history[-3:]) / 3.0
            earlier_trend = sum(flow_magnitude_history[-6:-3]) / 3.0 if len(flow_magnitude_history) >= 6 else recent_trend
            
            if recent_trend > earlier_trend * 1.5 and recent_trend > 500.0:
                print(f"\n⚠️  Flow magnitude is increasing rapidly (recent: {recent_trend:.1f}, earlier: {earlier_trend:.1f})")
                print("Consider reducing learning rate or increasing flow regularization.")
        
        # After the loop for all pairs is done
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # Check for NaN parameters at the end of each epoch
        planner_nan = check_and_fix_model_state(planner, "planner")
        actor_nan = check_and_fix_model_state(actor, "actor")
        critic_nan = check_and_fix_model_state(critic, "critic")
        target_nan = check_and_fix_model_state(target_critic, "target_critic")

        if planner_nan or actor_nan or critic_nan or target_nan:
            print(f"WARNING: Model parameters were reset due to NaN at epoch {epoch+1}")
            # If the critic was reset, re-sync the target critic
            if critic_nan:
                target_critic.load_state_dict(critic.state_dict())
        
        # --- MEMORY-EFFICIENT VALIDATION STEP at the end of each epoch ---
        print(f"\n📊 Running memory-efficient validation for epoch {epoch+1}...")
        
        # Memory optimization: Clear GPU cache before validation
        torch.cuda.empty_cache()
        gc.collect()
        
        # ADAPTIVE VALIDATION: Use fewer samples during training, more for final validation
        max_val_samples = 5 if epoch < epochs - 10 else 10  # Use more samples for final epochs
        
        # Run validation with memory optimizations
        current_val_mind = validate_memory_efficient(
            planner=planner,
            actor=actor,
            critic=critic, 
            val_loader=val_loader, 
            device=device,
            vol_shape=vol_shape,
            feature_shape=feature_shape,
            max_steps=max_steps,
            max_validation_samples=max_val_samples,
            use_raw_images=use_raw_images,
            dino_encoder=dino_encoder,
            pca_transformer=pca_transformer,
            dataset_type=dataset_type
        )
        
        # critic_scheduler.step(current_val_mind)
        # actor_scheduler.step(current_val_mind)
        # critic_lr = critic_scheduler.get_last_lr()[0]
        # actor_lr = actor_scheduler.get_last_lr()[0]
        
        # Just step the main schedulers
        planner_main_scheduler.step()
        critic_main_scheduler.step()

        # You can get the current LR for logging like this:
        current_lr_planner = planner_optimizer.param_groups[0]['lr']
        current_lr_critic = critic_optimizer.param_groups[0]['lr']

        # Clean up after validation
        torch.cuda.empty_cache()
        
        # Force garbage collection to clean up any easy-to-find objects
        gc.collect() 

        # Enhanced logging
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss_per_update, avg_reward, avg_flow_magnitude, current_val_mind, epoch_update_count, current_alpha, 
                             avg_total_loss, avg_planner_loss, avg_critic_loss, avg_alpha_loss, avg_decoder_loss, avg_sim_cost, avg_smooth_penalty, 
                             avg_mag_penalty, avg_refinement_improvement, current_lr_critic, current_lr_planner])

        # Print comprehensive epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary ({epoch_duration/60:.2f} min):")
        print(f"  LRs -> Critic: {current_lr_critic:.2e}, Planner: {current_lr_planner:.2e}")
        print(f"  Avg Loss: {avg_loss_per_update:.4f}")
        print(f"  Avg Reward: {avg_reward:.4f}")
        print(f"  Avg Flow Magnitude: {avg_flow_magnitude:.4f}")
        print(f"  Val MIND: {current_val_mind:.4f}")
        print(f"  Updates: {epoch_update_count}")
        print(f"  Avg Smooth Penalty: {avg_smooth_penalty:.4f}")
        print(f"  Avg Mag Penalty: {avg_mag_penalty:.4f}")
        print(f"  Current Alpha: {current_alpha:.4f}")
        print(f"  Avg Critic Loss: {avg_critic_loss:.4f}")
        print(f"  Avg Alpha Loss: {avg_alpha_loss:.4f}")
        
        # Get the current process's memory usage
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mem_gb = mem_info.rss / (1024 ** 3)  # Resident Set Size in Gigabytes
        vms_mem_gb = mem_info.vms / (1024 ** 3)  # Virtual Memory Size in Gigabytes

        print(f"  🧠 Memory Usage: RSS={rss_mem_gb:.2f} GB | VMS={vms_mem_gb:.2f} GB")

        # Check if this is the best model so far and save it; early stopping
        if current_val_mind < best_val_mind:
            best_val_mind = current_val_mind
            print(f"   💾 New best model saved! Val MIND: {best_val_mind:.4f}")
            patience_counter = 0  # Reset patience because we found a better model
            
            # Create a comprehensive checkpoint dictionary
            checkpoint = {
                'epoch': epoch + 1,
                'best_val_mind': best_val_mind,
                'loss': avg_loss_per_update,
                'planner_state_dict': planner.state_dict(),
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'target_critic_state_dict': target_critic.state_dict(),
                'planner_optimizer_state_dict': planner_optimizer.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
                # Schedulers are usually not critical to save, but you can add them
            }
            
            torch.save(checkpoint, os.path.join(save_dir, "best_vlearn_model.pth"))
        else:
            patience_counter += 1  # Increment patience because performance did not improve
            print(f"   ⏳ No improvement. Patience: {patience_counter}/{patience}")
        
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'best_val_mind': best_val_mind,
                'loss': avg_loss_per_update,
                'planner_state_dict': planner.state_dict(),
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'target_critic_state_dict': target_critic.state_dict(),
                'planner_optimizer_state_dict': planner_optimizer.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path}")
            
            print(f"   ⚡ Performing a hard update on the target network at epoch {epoch + 1}.")
            target_critic.load_state_dict(critic.state_dict())
        
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {patience} epochs without improvement.")
            break  # Exit the training loop
        
        # print("\n--- Searching for leaked tensor from the last update step ---")
        gc.collect() # Force garbage collection
        
        # find_leaking_tensors(epoch)
        # gc.collect() # Force garbage collection to clean up detached objects
        # print(f"\n--- 🕵️ OBJGRAPH MEMORY GROWTH ANALYSIS (End of Epoch {epoch + 1}) ---")
        # objgraph.show_growth()
        # print("-" * 60)
          
        
        # --- FINAL MEMORY LEAK ANALYSIS with tracemalloc ---
        print("\n" + "="*50)
        print(f"🔬 TRACEMALLOC ANALYSIS (End of Epoch {epoch+1})")
        print("="*50)

        # Take a new snapshot of the current memory usage
        snapshot2 = tracemalloc.take_snapshot()
 
        if snapshot1 is not None:
            # If we have a previous snapshot, compare it to the new one
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
            print("Top 10 memory differences since last epoch (potential leaks):")
            for stat in top_stats[:10]:
                print(stat)

        # Update the baseline snapshot for the next comparison
        snapshot1 = snapshot2
        
    # Save final model
    torch.save({
                'epoch': epoch + 1,
                'best_val_mind': best_val_mind,
                'loss': avg_loss_per_update,
                'planner_state_dict': planner.state_dict(),
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'target_critic_state_dict': target_critic.state_dict(),
                'planner_optimizer_state_dict': planner_optimizer.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
            }, os.path.join(save_dir, "final_vlearn_model.pth"))
    print(f"\n🏁 Training completed!")
    print(f"   Best validation Dice: {best_val_mind:.4f}")
    print(f"   Models saved in: {save_dir}")
    
    return policy