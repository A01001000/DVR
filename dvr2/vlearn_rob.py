from email import policy
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
compile_cache_dir = "/work/tc067/tc067/s2749465/torch_compile_cache"
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
from dvr2.models_rob import VLearnPolicy_Medium, SpatialTransformer, AdaptiveFlowPolicy_Medium
from dvr2.utils import MINDLoss, get_warped_grid, save_slice_as_png, inspect_actor_layers, apply_flow_scaling, get_effective_spacing
from dvr2.extract_dino import extract_single_dino_features, save_feature_slice_as_png
from dvr2.reward import HierarchicalRewardSystem, RewardNormalizer, compute_smoothness_penalty, normalize_advantages, multi_scale_reward, DualChannelRewardSystem, SimplifiedSSDRewardSystem, MindSSDRewardSystem

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
        # This is a standard and stable practice for multi-channel inputs.
        f_mr_norm = (f_mr_warped - f_mr_warped.mean()) / (f_mr_warped.std() + 1e-8)
        f_ct_norm = (f_ct_fixed - f_ct_fixed.mean()) / (f_ct_fixed.std() + 1e-8)
        
        # Flow is already on a reasonable scale, just divide by a constant
        flow_norm = flow_accumulated / flow_scale
        
        # Concatenate the consistently scaled components
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

def run_registration_episode(policy, target_policy, transformer, f_mr_3d, f_ct_3d,
                             replay_buffer, optimizers, scaler, reward_normalizer,
                             epoch, epochs, current_max_steps, total_steps, hyperparams, current_spacing=None):
    """
    Runs a complete registration episode for a single image pair in an isolated scope.
    Returns a list of experiences to be added to the replay buffer.
    """
    episode_metrics = {
        'total_reward': 0, 'reward_count': 0, 'total_sim_reward': 0, 'coarse_sim': 0, 
        'fine_sim': 0, 'smooth_penalty': 0, 'mag_penalty': 0, 'total_loss': 0,
        'actor_loss': 0, 'critic_loss': 0, 'alpha_loss': 0, 'update_count': 0
    }
    
    n_step = 3  # A common value, hyperparameter to tune
    n_step_buffer = deque(maxlen=n_step)
    
    # --- This is your existing episode loop ---
    flow_acc = torch.zeros((1, 3, *hyperparams['vol_shape']), device=f_mr_3d.device)

    for step in range(current_max_steps):
        # feature map shape: spatial dimensions (D, H, W) of the feature map
        feature_map_spatial_dims = f_mr_3d.shape[1:]  # e.g., [D, 16, 16]
                
        # WARP CURRENT FEATURE MAP
        with torch.no_grad():
            # DOWNSAMPLE THE FLOW to match the feature map resolution
            flow_for_features = F.interpolate(flow_acc, size=feature_map_spatial_dims, mode='trilinear', align_corners=False)
            flow_for_features = apply_flow_scaling(flow_for_features, flow_acc.shape[2:], feature_map_spatial_dims, current_spacing)
            
            # ------ DEBUG
            # test_flow = torch.ones_like(flow_for_features) * 5.0  # 5 voxel displacement
            # test_flow_norm = test_flow / (torch.tensor(f_mr_3d.shape[1:]).float().to(test_flow.device).view(1, 3, 1, 1, 1) / 2.0)

            # print(f"Test flow: {test_flow[0, :, 0, 0, 0]}")
            # print(f"Normalized test flow: {test_flow_norm[0, :, 0, 0, 0]}")
            # ------ DEBUG
                    
            # Upsample flow to match feature map dimensions if needed
            warped_f_mr_3d = transformer(f_mr_3d.unsqueeze(0), flow_for_features).squeeze(0)
                     
            # CREATE CURRENT STATE with proper normalization
            current_state = create_registration_state(
                f_mr_warped=warped_f_mr_3d,
                f_ct_fixed=f_ct_3d, 
                flow_accumulated=flow_for_features.squeeze(0),
                flow_scale=10.0  # Adjust based on your typical flow magnitudes
            )
        # GET ACTION from the policy based on the current warped state
        action_dist = policy.get_action_dist(current_state.unsqueeze(0))  # Add batch dimension
        incremental_flow_low = action_dist.sample().squeeze(0)  
        
        # --- EXPLORATION FIX: Add noise to the action ---
        # This encourages the agent to explore actions around its predicted mean.
        initial_noise_scale = 0.5
        final_noise_scale = 0.05
        noise_decay_steps = 50000
        current_noise_scale = initial_noise_scale * (final_noise_scale / initial_noise_scale) ** (total_steps / noise_decay_steps)
        noise = torch.randn_like(incremental_flow_low) * current_noise_scale
        action_with_noise = incremental_flow_low + noise
        
        print(f"Exploration noise scale: {current_noise_scale:.4f}")

        # Make sure to SQUEEZE AFTER adding noise
        incremental_flow_low = action_with_noise.squeeze(0)
                    
        # Calculate the entropy of the action distribution
        # This is our exploration bonus
        entropy = action_dist.entropy().mean()

        # UPSAMPLE the incremental flow action to the full image resolution
        incremental_flow_full_res = F.interpolate(
            incremental_flow_low.unsqueeze(0),
            size=flow_acc.shape[2:],
            mode='trilinear',
            align_corners=False
        ).squeeze(0)
        incremental_flow_full_res = apply_flow_scaling(incremental_flow_full_res, incremental_flow_low.shape[1:], flow_acc.shape[2:])
                
        # APPLY ACTION  (Clip flow to prevent extreme deformations)
        flow_update = flow_acc + incremental_flow_full_res
        new_flow_acc = hyperparams["MAX_FLOW_MAGNITUDE"] * torch.tanh(flow_update / hyperparams["MAX_FLOW_MAGNITUDE"])
        # print(f"Flow magnitude: min={new_flow_acc.min().item():.3f}, max={new_flow_acc.max().item():.3f}, mean={new_flow_acc.abs().mean().item():.3f}")
        # Add this right after you calculate new_flow_acc
        # print(f"=== Flow Debugging ===")
        # print(f"Original volume shape: {hyperparams['vol_shape']}")  # Should be [192, 192, 192]
        # print(f"Feature map shape: {f_mr_3d.shape}")  # Should be [64, D, 16, 16]
        # print(f"flow_acc shape: {flow_acc.shape}")
        # print(f"flow_for_features shape: {flow_for_features.shape}")
        # print(f"Flow magnitude stats:")
        # print(f"flow_acc: mean_abs={flow_acc.abs().mean():.2f}, max_abs={flow_acc.abs().max():.2f}")
        # print(f"  - flow_for_features: mean_abs={flow_for_features.abs().mean():.2f}, max_abs={flow_for_features.abs().max():.2f}")

        # Check what this means in voxel displacement
        max_displacement_voxels = flow_acc.abs().max().item()
        print(f"Flow stats: mean_abs={flow_acc.abs().mean():.3f}, max_abs={flow_acc.abs().max():.3f}")
        print(f"Max displacement in voxels: {max_displacement_voxels:.2f}")
        print(f"Max displacement as % of volume: {(max_displacement_voxels / 192) * 100:.1f}%")

        # WARP IMAGE for reward calculation
        with torch.no_grad():
            # Downsample the new flow to calculate reward in feature space
            flow_for_reward = F.interpolate(new_flow_acc, size=f_mr_3d.shape[1:], mode='trilinear', align_corners=False)
            flow_for_reward = apply_flow_scaling(flow_for_reward, new_flow_acc.shape[2:], f_mr_3d.shape[1:], current_spacing)
            
            warped_feat = transformer(f_mr_3d.unsqueeze(0), flow_for_reward).squeeze(0)
                            
            # smooth_val = compute_smoothness_penalty(new_flow_acc).item()
            # mag_val = (new_flow_acc**2).mean().item()
            
            # Use the new hierarchical reward:
            reward, similarity_reward, smooth_val, mag_val = hyperparams["reward_system"].get_reward(
                warped_features=warped_feat, 
                fixed_features=f_ct_3d, 
                flow_field=new_flow_acc, 
                smoothness_weight=hyperparams["SMOOTHNESS_WEIGHT"], 
                magnitude_weight=hyperparams["MAGNITUDE_WEIGHT"], 
                similarity_weight=hyperparams["SIMILARITY_WEIGHT"]
            )
            coarse_reward = 0
            fine_reward = 0
            coarse_weight = 0
            reward = np.clip(reward, -5.0, 5.0) 
            reward_normalizer.update(reward)
                        
            if step % 10 == 0:  # Log every 10 steps
                print(f"   Rewards - Total: {reward:.4f}, Coarse: {coarse_reward:.4f} (w={coarse_weight:.2f}), "
                    f"Fine: {fine_reward:.4f}")
                    
        done = (step == current_max_steps - 1)
                    
        # Track metrics
        episode_metrics['total_reward'] += reward
        episode_metrics['reward_count'] += 1
        episode_metrics['total_sim_reward'] += similarity_reward
        episode_metrics['coarse_sim'] += coarse_reward
        episode_metrics['fine_sim'] += fine_reward
        episode_metrics['smooth_penalty'] += smooth_val
        episode_metrics['mag_penalty'] += mag_val

        # GET NEXT STATE by warping the feature map with the NEW flow
        with torch.no_grad():
            # Downsample the NEW flow field before creating the next state
            flow_for_next_features = F.interpolate(new_flow_acc, size=f_mr_3d.shape[1:], mode='trilinear', align_corners=False)
            flow_for_next_features = apply_flow_scaling(flow_for_next_features, new_flow_acc.shape[2:], f_mr_3d.shape[1:], current_spacing)
            
            # Warp the original moving features with the NEW flow
            next_warped_f_mr_3d_next = transformer(f_mr_3d.unsqueeze(0), flow_for_next_features).squeeze(0)
    
            # Create the next state, ensuring it also has 131 channels
            next_state = create_registration_state(
                f_mr_warped=next_warped_f_mr_3d_next,
                f_ct_fixed=f_ct_3d, 
                flow_accumulated=flow_for_next_features.squeeze(0),
                flow_scale=10.0  # Adjust based on your typical flow magnitudes
            )
                        
        # Store the current transition in the temporary buffer
        experience = (
            to_uint8_numpy(current_state.clone().detach().cpu()),
            to_uint8_numpy(action_dist.mean.clone().detach().cpu()),
            to_uint8_numpy(action_dist.stddev.clone().detach().cpu()),
            reward, 
            to_uint8_numpy(next_state.clone().detach().cpu()),
            float(done),
            to_uint8_numpy(incremental_flow_low.clone().detach().cpu())
        )
        n_step_buffer.append(experience)
        
        # If the buffer is full, we can calculate the N-step return
        if len(n_step_buffer) == n_step:
            # Get the rewards and the final next_state from the buffer
            rewards = [transition[3] for transition in n_step_buffer]

            # Calculate the discounted sum of rewards (the N-step return)
            n_step_reward = 0
            for i in range(n_step):
                n_step_reward += (hyperparams['GAMMA'] ** i) * rewards[i]
                
            # The experience we store is from the FIRST step in the buffer,
            # but with the new N-step reward and the FINAL next_state.
            first_state, first_mean, first_std, _, _, _, first_action = n_step_buffer[0]
            final_next_state = n_step_buffer[-1][4]
            is_terminal = n_step_buffer[-1][5]

            # STORE IN BUFFER
            # We store the state that led to the action, and the resulting next_state
            experience = (
                first_state, 
                first_mean, 
                first_std,
                n_step_reward,  # Use the calculated N-step reward
                final_next_state,
                is_terminal,
                first_action
            )
            replay_buffer.push_preformatted(experience)
            
            del first_state, first_mean, first_std, first_action, final_next_state
            del rewards, experience, n_step_reward
                    
        flow_acc = new_flow_acc.detach()
        total_steps += 1
                    
        del feature_map_spatial_dims, flow_for_features, warped_f_mr_3d, current_state, action_dist
        del incremental_flow_low, noise, action_with_noise, entropy, incremental_flow_full_res     
        del flow_update, new_flow_acc
        del flow_for_reward, warped_feat, smooth_val, mag_val, similarity_reward        
        del reward, coarse_reward, fine_reward, coarse_weight
        del flow_for_next_features, next_warped_f_mr_3d_next, next_state
        
        # --- Optimization Step (The Core of V-Learn) ---
        # V-Learn Paper Key Innovation:
        # Apply importance sampling to the ENTIRE Bellman error (loss function)
        # rather than just the Bellman targets. This provides:
        # 1. Lower variance than V-trace and other methods
        # 2. Better stability for off-policy V-function learning
        # 3. Upper bound on naive Bellman error via Jensen's inequality
                    
        if len(replay_buffer) >= hyperparams['BATCH_SIZE'] and total_steps % hyperparams['UPDATE_EVERY_N_STEPS'] == 0:
            # print(f"🔥 PERFORMING UPDATE #{epoch_update_count + 1} at step {total_steps} (epoch {epoch+1})")
            # Sample a batch of experiences from the buffer
            s, old_a_mean, old_a_std, r, s_prime, d, a = replay_buffer.sample(hyperparams['BATCH_SIZE'])

            if not torch.isfinite(s_prime).all():
                print("NaN or Inf detected in s_prime!")
            if not torch.isfinite(old_a_mean).all():
                print("NaN or Inf detected in old_a_mean!")
            if not torch.isfinite(old_a_std).all():
                print("NaN or Inf detected in old_a_std!")
                            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # --- Twin Critic Value Calculation (V-Learn paper) ---
                # Use autocast for mixed-precision performance (helps speed up training)
                with torch.no_grad():
                    r_normalized = reward_normalizer.normalize(r)
                    # Get both critic values from target network
                    next_value1, next_value2 = target_policy.forward_critics(s_prime)
                    # Use minimum for target (reduces overestimation bias)
                    next_value = torch.min(next_value1, next_value2)
                    reward_clip_range = 5.0 # A reasonable starting point
                    next_value_clipped = torch.clamp(next_value, -reward_clip_range, reward_clip_range)
                                
                    # Use the clipped value to calculate the final target
                    value_target = r_normalized.view(-1, 1) + hyperparams["GAMMA"] * next_value_clipped * (1 - d.view(-1, 1))

                # Get current values from both critics
                current_value1, current_value2 = policy.forward_critics(s)
                
                # --- Actor Policy Calculation ---
                # 1. Get the new action distribution from the current policy
                new_dist_critic = policy.get_action_dist(s)

                # 2. Reconstruct the old action distribution using parameters from the replay buffer
                # This is where you use old_a_mean and old_a_std from the buffer
                old_dist_critic = torch.distributions.Normal(old_a_mean, old_a_std)
    
                # Get the log probability of that action under both distributions
                new_log_p = new_dist_critic.log_prob(a).mean(dim=(1,2,3,4)).view(-1, 1)
                old_log_p = old_dist_critic.log_prob(a).mean(dim=(1,2,3,4)).view(-1, 1)
    
                ON_POLICY_WARMUP_STEPS = 5000 # Number of updates to treat as on-policy

                if total_steps < ON_POLICY_WARMUP_STEPS:
                    # During warmup, the policy is changing too fast for IS.
                    # Treat it as on-policy by setting rho = 1.
                    # rho = torch.ones(new_log_p.size(0), 1, device=hyperparams["device"])
                    warmup_factor = total_steps / ON_POLICY_WARMUP_STEPS
                    log_rho = (new_log_p - old_log_p.detach()) * warmup_factor
                    rho = torch.exp(torch.clamp(log_rho, math.log(0.1), math.log(5.0)))
                    rho = rho / (rho.mean().detach() + 1e-8)
                else:
                    # After warmup, the policy is more stable, so we can use RIS.
                    # log-space RIS denominator: log( β·π + (1-β)·b )
                    log_beta = math.log(hyperparams["BETA"])
                    log_1mb  = math.log(1 - hyperparams["BETA"])
                    # logsumexp for numerical stability
                    den = torch.logsumexp(torch.stack([log_beta + new_log_p, log_1mb + old_log_p.detach()], dim=0), dim=0)
                    log_rho = new_log_p - den
                    # bound rho to avoid zeroing the loss; keep gradients
                    rho = torch.exp(torch.clamp(log_rho, math.log(0.8), math.log(1.2)))  # [0.1, 2.0]
                    # optional: normalize per-batch to mean ≈1
                    rho = rho / (rho.mean().detach() + 1e-8)

                    rho = rho.view(-1,1)  # match target shape
                                
                    del log_beta, log_1mb, den, log_rho
                        
                # --- Calculate Twin Critic Losses (V-Learn Paper Approach) ---
                # V-Learn key innovation: Apply importance weights to entire Bellman error
                # Calculate MSE loss without reduction to get per-sample losses for both critics
                # V-Learn approach: Apply importance weights to the loss function (not just targets)
                # Using rho_clipped WITHOUT detach to maintain gradient flow as per V-Learn paper
                        
                if torch.isnan(s).any() or torch.isnan(s_prime).any():
                    print("🔥🔥🔥 DEBUG: NaN found in input states! Aborting update.")
                    continue # Skip this update entirely

                # ----- CORRECTED VLEARN CRITIC LOSS: Apply rho before the final mean -----
                # This preserves spatial information during weighting.
                critic1_loss = F.smooth_l1_loss(current_value1, value_target, reduction='none')
                critic2_loss = F.smooth_l1_loss(current_value2, value_target, reduction='none')
                critic_loss = (critic1_loss * rho + critic2_loss * rho).mean()
                
                # V-Learn Actor Update Logic (Pure V-Learn)
                with torch.no_grad():
                    # Use minimum of twin critics for advantage calculation (reduces overestimation)
                    # current_value_min_detached = torch.min(current_value1.detach(), current_value2.detach())
                    advantage = r_normalized.view(-1, 1) - torch.min(current_value1, current_value2).detach()
                                # print(f"advantage mean before normalization={advantage.mean().item():.6f}, std={advantage.std().item():.6f}")

                    # Advantage Normalization (V-Learn paper technique)
                    advantage = normalize_advantages(advantage)
                    advantage = torch.clamp(advantage, -5.0, 5.0)
                                                    
                # Get the new distribution
                new_dist_actor = policy.get_action_dist(s)
                old_dist_actor = torch.distributions.Normal(old_a_mean, old_a_std)
                            
                # action 'a' that was actually taken, which is stored in the buffer as old_a_mean (deterministic action that was sampled and stored; since we took the mean during sampling in the past)
                # pi: log_prob of that action under the NEW policy (reconstruct the old action 'a' from the buffer)
                new_log_p_actor = new_dist_actor.log_prob(a).sum(dim=(1,2,3,4)).view(-1, 1)

                # b: log_prob of that action under the OLD policy
                old_log_p_actor = old_dist_actor.log_prob(a).sum(dim=(1,2,3,4)).view(-1, 1)

                # Get current alpha value
                alpha = hyperparams["log_alpha"].exp()
                            
                # Calculate entropy and alpha loss
                entropy = new_dist_actor.entropy().mean()
                alpha_loss = -(alpha * (entropy.detach() + hyperparams["target_entropy"])).mean()
                            
                # Calculate final actor loss
                # Calculate the importance ratio: standard policy gradient loss using a clipped objective, similar to PPO. 
                clip_ratio = 0.2 # Standard PPO hyperparameter
                log_ratio_pi_b = old_log_p_actor.detach() - new_log_p_actor
                log_ratio_pi_b = torch.clamp(log_ratio_pi_b, -10.0, 10.0)
                ratio = 1.0 / (hyperparams["BETA"] + (1.0 - hyperparams["BETA"]) * torch.exp(log_ratio_pi_b))

                # Calculate the first surrogate objective
                surr1 = ratio * advantage.detach()
                # Calculate the second, "clipped" surrogate objective
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage.detach()

                # The final loss is the minimum of the two, plus the entropy bonus.
                # This prevents the updates from ever being too large.
                actor_loss = -torch.min(surr1, surr2).mean() - alpha.detach() * entropy
    
                total_loss = critic_loss + actor_loss + alpha_loss
        
                episode_metrics['total_loss'] += total_loss.item()
                episode_metrics['actor_loss'] += actor_loss.item()
                episode_metrics['critic_loss'] += critic_loss.item()
                episode_metrics['alpha_loss'] += alpha_loss.item()

                print(f"   💡 Update #{episode_metrics['update_count']}: total_loss={total_loss.item():.6f}, critic_loss={critic_loss.item():.6f}, actor_loss={actor_loss.item():.6f}, alpha_loss={alpha_loss.item():.6f}, epoch_total_loss={episode_metrics['total_loss']:.6f}")

                # --- Backpropagation with Gradient Accumulation ---
                # Check for NaN in loss before backpropagation
                if not torch.isfinite(total_loss):
                    print(f"WARNING: Invalid loss detected (NaN or Inf)! Skipping update.")
                    continue

            # --- Backpropagation ---
            total_loss_scaled = total_loss / hyperparams["GRADIENT_ACCUMULATION_STEPS"]
            scaler.scale(total_loss_scaled).backward()
            
            del r_normalized, next_value1, next_value2, next_value, next_value_clipped, value_target
            del current_value1, current_value2, new_dist_critic, old_dist_critic, new_log_p, old_log_p
            del rho, critic1_loss, critic2_loss, critic_loss, total_loss, total_loss_scaled
            del advantage, new_dist_actor, old_dist_actor, new_log_p_actor, old_log_p_actor
            del alpha, entropy, actor_loss, alpha_loss, ratio, surr1, surr2, log_ratio_pi_b

            if (total_steps % hyperparams["GRADIENT_ACCUMULATION_STEPS"]) == 0:
                # We have processed enough micro-batches, now perform the actual model update

                # Always update the critic
                scaler.unscale_(optimizers["critic"])
                scaler.unscale_(optimizers["actor"])
                scaler.unscale_(optimizers["alpha"])

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)

                scaler.step(optimizers["critic"])
                scaler.step(optimizers["alpha"]) # Step the alpha optimizer

                # 4. The "Bulletproof" Actor Update
                # directly check if a parameter that the actor_optimizer is responsible for has a non-None gradient to know if the actor participated in the backward pass.
                if episode_metrics['update_count'] % 2 == 0 and policy.flow_mean.weight.grad is not None:
                    scaler.step(optimizers["actor"])

                # 5. Update the scaler once at the end
                scaler.update()
                            
                # Zero the gradients for the NEXT accumulation cycle
                optimizers["critic"].zero_grad()
                optimizers["actor"].zero_grad()
                optimizers["alpha"].zero_grad()

                episode_metrics['update_count'] += 1
                        
                # 6. Check for NaN parameters after update and reset if found
                check_and_fix_model_state(policy, "policy")
                check_and_fix_model_state(target_policy, "target_policy")
                        
                # --- Update Target Network ---
                with torch.no_grad():
                    for param, target_param in zip(policy.parameters(), target_policy.parameters()):
                        target_param.data.mul_(hyperparams["POLYAK_TAU"])
                        target_param.data.add_((1 - hyperparams["POLYAK_TAU"]) * param.data)
                  
                # If the episode ended due to the penalty, break the inner loop
                if done:
                    break
                
    if len(n_step_buffer) > 0:
        final_next_state_numpy = n_step_buffer[-1][4]
        min_val, max_val = -5.0, 5.0
        final_next_state_float = final_next_state_numpy.astype(np.float32) / 255.0
        dequantized = final_next_state_float * (max_val - min_val) + min_val
        
        state_tensor = torch.from_numpy(dequantized).unsqueeze(0).to(hyperparams["device"])
        final_v_next = target_policy.get_value(state_tensor)

        # THE FIX: Detach the tensor to remove its computation graph
        G = final_v_next.detach()
    else:
        # Handle cases where the episode ends with an empty buffer
        G = torch.tensor(0.0, device=hyperparams["device"])
    
    while len(n_step_buffer) > 0:
        # Calculate the return for the current buffer contents
        for transition in reversed(list(n_step_buffer)):
            reward = transition[3]
            G = reward + hyperparams['GAMMA'] * G

        # The experience we store is from the FIRST step in the buffer
        first_state, first_mean, first_std, _, _, is_terminal, first_action = n_step_buffer[0]
        
        # We use the calculated G as the "n_step_reward"
        n_step_reward = G.item()
        
        # The next_state is from the last element
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
        
        # Remove the transition we just processed
        n_step_buffer.popleft()
        
        del first_state, first_mean, first_std, is_terminal, first_action, n_step_reward, final_next_state, experience

    del n_step_buffer, replay_buffer, f_mr_3d, f_ct_3d, reward_normalizer, final_v_next, state_tensor, done, G

    return flow_acc, total_steps, episode_metrics

def train_vlearn(dino_encoder, pca_transformer, feature_loader, val_loader, device, epochs=100, max_steps=5, save_dir="checkpoints", dino_dir=None, is_continue_training=False):
    """
    Train V-Learn policy with configurable network architectures
    """
    torch.autograd.set_detect_anomaly(True)
    
    # --- V-Learn Paper Hyperparameters with Memory-Safe Options ---
    IMG_SIZE = 128
    GAMMA = 0.99  # Discount factor for future rewards
    POLYAK_TAU = 0.001 # Target network update rate
    vol_shape = [128, 128, 128]
    feature_shape = [64, 128, 16, 16]
    C_feat = 64
    
    BATCH_SIZE = 8 # Memory-optimized buffer capacity based on network size
    BUFFER_CAPACITY = 10000  # V-Learn paper standard for large setups
    UPDATE_EVERY_N_STEPS = 4  # More frequent updates to prevent staleness
    
    # COMPENSATION STRATEGIES for reduced batch sizes:
    # 1. Accumulate gradients to simulate larger effective batch size
    GRADIENT_ACCUMULATION_STEPS = max(1, 64 // BATCH_SIZE)  # Aim for effective batch ~64
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    
    # 2. Scale learning rate based on EFFECTIVE batch size (not physical batch size)
    
    MAX_FLOW_MAGNITUDE = 10.0
    # LEARNING ENCOURAGEMENT: New weight for progress-based reward
    IMPROVEMENT_WEIGHT = 50.0 
    SIMILARITY_WEIGHT = 50.0
    NGF_WEIGHT = 0.15
    BETA = 0.9
    
    # PENALTY SCHEDULE
    PENALTY_WARMUP_EPOCHS = 15      # Use low penalties for the first 15 epochs
    PENALTY_ANNEALING_EPOCHS = 75   # Gradually increase penalties until epoch 75
    SMOOTHNESS_WEIGHT_START = 5.0   # The low value that worked well initially
    SMOOTHNESS_WEIGHT_END = 5.0     # The high value to enforce refinement later
    MAGNITUDE_WEIGHT_START = 2.0    # The low value that worked well initially
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
    
    # Create V-Learn policy with configurable architecture
    policy = AdaptiveFlowPolicy_Medium(in_channels=(C_feat * 2) + 3, use_checkpointing=True).to(device)
    
    # Target network for stable value estimation [cite: 278, 279]
    target_policy = AdaptiveFlowPolicy_Medium(in_channels=(C_feat * 2) + 3, use_checkpointing=True).to(device)
    target_policy.load_state_dict(policy.state_dict())
    target_policy.eval() # Target network is not trained directly
    
    transformer = SpatialTransformer().to(device)
    
    hierarchical_reward_system = HierarchicalRewardSystem(device)
    dual_reward_system = DualChannelRewardSystem(device)
    simplified_reward_system = SimplifiedSSDRewardSystem(device)
    mind_reward_system = MindSSDRewardSystem(device)

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"📊 Policy network parameters: {total_params:,}")

    # CRITIC PARAMETERS: All encoder layers + critic-specific layers
    critic_params = [
        *policy.critic_conv1.parameters(),
        *policy.critic_conv2.parameters(),
        *policy.critic_conv3.parameters(),
        *policy.critic1.parameters(),
        *policy.critic2.parameters()
    ]
    
    # ACTOR PARAMETERS: All decoder layers + flow heads
    actor_params = [
        *policy.encoder_conv1.parameters(),
        *policy.encoder_conv2.parameters(),
        *policy.encoder_conv3.parameters(),
        # *policy.encoder_conv4.parameters(), 
        *policy.decoder_conv1.parameters(),
        *policy.decoder_conv2.parameters(),
        *policy.decoder_conv3.parameters(),
        # *policy.decoder_conv4.parameters(), 
        *policy.flow_mean.parameters(),
        *policy.flow_log_std.parameters()
    ]

    critic_optimizer = Adam(critic_params, lr=LR_CRITIC_START)
    actor_optimizer = Adam(actor_params, lr=LR_ACTOR_START)
    # critic_scheduler = ReduceLROnPlateau(critic_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # actor_scheduler = ReduceLROnPlateau(actor_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    actor_scheduler = lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=epochs - WARMUP_EPOCHS, eta_min=1e-6)
    critic_scheduler = lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=epochs - WARMUP_EPOCHS, eta_min=1e-6)
    
    # Sequential scheduler to handle the warmup phase
    actor_warmup_scheduler = lr_scheduler.LinearLR(actor_optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    critic_warmup_scheduler = lr_scheduler.LinearLR(critic_optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)

    # Chain the warmup and cosine schedulers together
    actor_main_scheduler = lr_scheduler.SequentialLR(actor_optimizer, schedulers=[actor_warmup_scheduler, actor_scheduler], milestones=[WARMUP_EPOCHS])
    critic_main_scheduler = lr_scheduler.SequentialLR(critic_optimizer, schedulers=[critic_warmup_scheduler, critic_scheduler], milestones=[WARMUP_EPOCHS])
    
    # Adaptive Entropy (SAC-style)
    # The target entropy is a heuristic. A common choice is -|A|, the negative dimensionality of the action space.
    # Since your action space is a 3D flow field, we can approximate this.
    # Let's assume the low-res action space is 3x16x16x16 = 12288
    target_entropy = -10.0 # This is a heuristic value, can be tuned based on your action space
    # target_entropy = -np.prod((3, D_feat // 32, H_feat // 32, W_feat // 32)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = Adam([log_alpha], lr=3e-4)
    
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

    checkpoint_path = os.path.join(save_dir, "checkpoint_epoch_16.pth")
    log_path = os.path.join(save_dir, "training_log.csv")
    
    if os.path.exists(checkpoint_path) and is_continue_training:
        print(f"✅ Resuming training by loading weights from: {checkpoint_path}")
        
        # Load the dictionary from the file
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # --- Restore the States ---
        
        # Restore model weights
        policy.load_state_dict(checkpoint['policy_state_dict'])
        target_policy.load_state_dict(checkpoint['target_policy_state_dict'])
        
        # Restore optimizer states (important for momentum, etc.)
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Restore the training progress
        # We add 1 because we want to start the *next* epoch
        start_epoch = checkpoint['epoch'] 
        best_val_mind = checkpoint['best_val_mind']

        print(f"✅ Resumed successfully. Starting at Epoch {start_epoch + 1}.")
        
        # TODO: CRITICAL STEP: Do NOT load the old replay buffer. 
        # We are intentionally starting with a fresh one.
        
    else:
        # Enhanced logging
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Avg_Update_Loss", "Avg_Reward", "Avg_Flow_Magnitude", "Val_MIND", "Total_Updates_This_Epoch", "Current_Alpha", "Avg_Critic_Loss", "Avg_Actor_Loss", "Avg_Alpha_Loss", "Avg_Total_Coarse_Weight", "Avg_Coarse_Reward", "Avg_Fine_Reward", "Avg Smooth Penalty", "Avg Magnitude Penalty", "Critic Learning Rate", "Actor Learning Rate"])
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
    
    # Before your training loop begins
    print("🚀 Compiling model for a significant speed boost...")
    policy = torch.compile(policy)
    target_policy = torch.compile(target_policy)
    transformer = torch.compile(transformer)
    
    # Normalize rewards
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
    reward_normalizer = RewardNormalizer(num_inputs=1) 
    scaler = GradScaler()
    
    # Multiple resolutions for robustness
    resolutions = {"low": (64, 64, 64), "full": (128, 128, 128)}

    # Bundle optimizers and some hyperparams for easier passing
    optimizers = {'actor': actor_optimizer, 'critic': critic_optimizer, 'alpha': alpha_optimizer}
    hyperparams = {
        'BATCH_SIZE': BATCH_SIZE,
        'UPDATE_EVERY_N_STEPS': UPDATE_EVERY_N_STEPS,
        'GAMMA': GAMMA,
        'POLYAK_TAU': POLYAK_TAU,
        'BETA': BETA,
        'target_entropy': target_entropy,
        'GRADIENT_ACCUMULATION_STEPS': GRADIENT_ACCUMULATION_STEPS,
        'reward_system': mind_reward_system,
        'SIMILARITY_WEIGHT': SIMILARITY_WEIGHT,
        'SMOOTHNESS_WEIGHT': SMOOTHNESS_WEIGHT_START,
        'MAGNITUDE_WEIGHT': MAGNITUDE_WEIGHT_START,
        'device': device,
        'log_alpha': log_alpha, 
        'MAX_FLOW_MAGNITUDE': MAX_FLOW_MAGNITUDE
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

        policy.set_epoch(epoch)
        target_policy.set_epoch(epoch)
        policy.train()
        
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
        """

        if epoch < SIMILARITY_PHASE_EPOCHS:
            # Phase 1: Teach the agent ONLY how to align images.
            # Penalties are turned off.
            hyperparams["similarity_weight"] = 50.0
            hyperparams["smoothness_weight"] = 0.0
            hyperparams["magnitude_weight"] = 0.0
            if epoch == 0:
                print("\n--- CURRICULUM PHASE 1: SIMILARITY ONLY ---")
        else:
            # Phase 2: Now that it knows how to align, teach it to be smooth.
            # Turn on the penalties.
            hyperparams["similarity_weight"] = 50.0
            hyperparams["smoothness_weight"] = 5.0
            hyperparams["magnitude_weight"] = 2.0
            if epoch == SIMILARITY_PHASE_EPOCHS:
                print("\n--- CURRICULUM PHASE 2: INTRODUCING PENALTIES ---")
        
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
            
        vol_shape = current_resolution
        D_vol, H_vol, W_vol = vol_shape
        hyperparams['vol_shape'] = vol_shape

        # The feature_loader provides batches of pre-computed feature tensors
        for batch_idx, (mr_batch, ct_batch, batch_spacing) in enumerate(tqdm(feature_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            mr_batch, ct_batch = mr_batch.to(device), ct_batch.to(device)
            batch_size = mr_batch.size(0)
            
            # on the fly downsampling for diff resolutions 
            if current_resolution != resolutions["full"]:
                mr_batch = F.interpolate(mr_batch, size=current_resolution, mode='trilinear')
                ct_batch = F.interpolate(ct_batch, size=current_resolution, mode='trilinear')
                
                # Update effective spacing
                batch_spacing = [get_effective_spacing(sp, resolutions["full"], current_resolution) for sp in batch_spacing]

            # Extract features ON THE FLY using the student model
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
                    
                    f_mr_3d = mixed_f_mr[i] # Use the mixed features for the state
                    f_ct_3d = mixed_f_ct[i]
                    
                    current_spacing = batch_spacing[i]
                
                flow_acc = torch.zeros((1, 3, D_vol, H_vol, W_vol), device=device)

                # --- Registration Episode & Optimization Loop ---
                flow_acc, total_steps, episode_metrics = run_registration_episode(
                    policy, target_policy, transformer, f_mr_3d, f_ct_3d,
                    replay_buffer, optimizers, scaler, reward_normalizer,
                    epoch, epochs, current_max_steps, total_steps, hyperparams, current_spacing
                )

                per_voxel_magnitudes = torch.norm(flow_acc.detach(), p=2, dim=1) # Calculate magnitude along channel dim
                epoch_total_flow_magnitude += per_voxel_magnitudes.mean().item()
                epoch_sample_count += 1
                epoch_total_reward += episode_metrics['total_reward']
                epoch_reward_count += episode_metrics['reward_count']   
                epoch_total_sim_reward += episode_metrics['total_sim_reward']
                epoch_total_coarse_sim += episode_metrics['coarse_sim']
                epoch_total_fine_sim += episode_metrics['fine_sim']
                epoch_smooth_penalty += episode_metrics['smooth_penalty']
                epoch_mag_penalty += episode_metrics['mag_penalty']
                epoch_total_loss += episode_metrics['total_loss']
                epoch_actor_loss += episode_metrics['actor_loss']
                epoch_critic_loss += episode_metrics['critic_loss']
                epoch_alpha_loss += episode_metrics['alpha_loss']
                epoch_update_count += episode_metrics['update_count']

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
        avg_smooth_penalty = epoch_smooth_penalty / epoch_reward_count if epoch_reward_count > 0 else 0
        avg_mag_penalty = epoch_mag_penalty / epoch_reward_count if epoch_reward_count > 0 else 0
        avg_critic_loss = epoch_critic_loss / epoch_update_count if epoch_update_count > 0 else 0
        avg_actor_loss = epoch_actor_loss / epoch_update_count if epoch_update_count > 0 else 0
        avg_alpha_loss = epoch_alpha_loss / epoch_update_count if epoch_update_count > 0 else 0
        current_alpha = log_alpha.exp().item()
        avg_total_sim_reward =  epoch_total_sim_reward / epoch_reward_count if epoch_reward_count > 0 else 0
        avg_total_coarse_sim = epoch_total_coarse_sim / epoch_reward_count if epoch_reward_count > 0 else 0
        avg_total_fine_sim = epoch_total_fine_sim / epoch_reward_count if epoch_reward_count > 0 else 0

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
        model_nan_found = check_and_fix_model_state(policy, "policy")
        target_nan_found = check_and_fix_model_state(target_policy, "target_policy")
        
        if model_nan_found or target_nan_found:
            print(f"WARNING: Model parameters were reset due to NaN at epoch {epoch+1}")
            # If we had to reset the model, sync target network
            if model_nan_found:
                target_policy.load_state_dict(policy.state_dict())
        
        # --- MEMORY-EFFICIENT VALIDATION STEP at the end of each epoch ---
        print(f"\n📊 Running memory-efficient validation for epoch {epoch+1}...")
        
        # Memory optimization: Clear GPU cache before validation
        torch.cuda.empty_cache()
        gc.collect()
        
        # ADAPTIVE VALIDATION: Use fewer samples during training, more for final validation
        max_val_samples = 5 if epoch < epochs - 10 else 10  # Use more samples for final epochs
        
        # Run validation with memory optimizations
        current_val_mind = validate_memory_efficient(
            policy=policy, 
            val_loader=val_loader, 
            device=device,
            vol_shape=vol_shape,
            feature_shape=feature_shape,
            max_steps=max_steps,
            max_validation_samples=max_val_samples
        )
        
        # critic_scheduler.step(current_val_mind)
        # actor_scheduler.step(current_val_mind)
        # critic_lr = critic_scheduler.get_last_lr()[0]
        # actor_lr = actor_scheduler.get_last_lr()[0]
        
        # Just step the main schedulers
        actor_main_scheduler.step()
        critic_main_scheduler.step()

        # You can get the current LR for logging like this:
        current_lr_actor = actor_optimizer.param_groups[0]['lr']
        current_lr_critic = critic_optimizer.param_groups[0]['lr']

        # Clean up after validation
        torch.cuda.empty_cache()
        
        # Force garbage collection to clean up any easy-to-find objects
        gc.collect() 

        # Enhanced logging
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss_per_update, avg_reward, avg_flow_magnitude, current_val_mind, epoch_update_count, current_alpha, 
                             avg_critic_loss, avg_actor_loss, avg_alpha_loss, avg_total_sim_reward, avg_total_coarse_sim, avg_total_fine_sim, avg_smooth_penalty, avg_mag_penalty, current_lr_critic, current_lr_actor])

        # Print comprehensive epoch summary
        print(f"\n🎯 Epoch {epoch+1}/{epochs} Summary ({epoch_duration/60:.2f} min):")
        print(f"  LRs -> Critic: {current_lr_critic:.2e}, Actor: {current_lr_actor:.2e}")
        print(f"   📉 Avg Loss: {avg_loss_per_update:.4f}")
        print(f"   🎁 Avg Reward: {avg_reward:.4f}")
        print(f"   📏 Avg Flow Magnitude: {avg_flow_magnitude:.4f}")
        print(f"   🎯 Val MIND: {current_val_mind:.4f}")
        print(f"   📚 Updates: {epoch_update_count}")
        print(f"   📏 Avg Smooth Penalty: {avg_smooth_penalty:.4f}")
        print(f"   📉 Avg Mag Penalty: {avg_mag_penalty:.4f}")
        print(f"   🔍 Current Alpha: {current_alpha:.4f}")
        print(f"   📊 Avg Critic Loss: {avg_critic_loss:.4f}")
        print(f"   📊 Avg Actor Loss: {avg_actor_loss:.4f}")
        print(f"   📊 Avg Alpha Loss: {avg_alpha_loss:.4f}")
        
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
                'policy_state_dict': policy.state_dict(),
                'target_policy_state_dict': target_policy.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
                # 'actor_scheduler_state_dict': actor_scheduler.state_dict(),
                # 'critic_scheduler_state_dict': critic_scheduler.state_dict()
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
                'policy_state_dict': policy.state_dict(),
                'target_policy_state_dict': target_policy.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
                # 'actor_scheduler_state_dict': actor_scheduler.state_dict(),
                # 'critic_scheduler_state_dict': critic_scheduler.state_dict()
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path}")
            
            print(f"   ⚡ Performing a hard update on the target network at epoch {epoch + 1}.")
            target_policy.load_state_dict(policy.state_dict())
        
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
                'policy_state_dict': policy.state_dict(),
                'target_policy_state_dict': target_policy.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
                # 'actor_scheduler_state_dict': actor_scheduler.state_dict(),
                # 'critic_scheduler_state_dict': critic_scheduler.state_dict()
            }, os.path.join(save_dir, "final_vlearn_model.pth"))
    print(f"\n🏁 Training completed!")
    print(f"   Best validation Dice: {best_val_mind:.4f}")
    print(f"   Models saved in: {save_dir}")
    
    return policy