import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint

class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src, flow=None, grid=None, mode='bilinear'):
        """
        src: (B,C,D,H,W) or (B,D,H,W)
        flow: (B,3,D,H,W) displacement field (voxels) [optional if grid is given]
        grid: (B,D,H,W,3) precomputed sampling grid [optional]
        """
        if src.ndim == 4:  # add channel dim
            src = src.unsqueeze(1)
        B, C, D, H, W = src.shape
        device = src.device

        # If precomputed grid is given, just use it
        if grid is not None:
            new_locs = grid
        else:
            # Otherwise build it (same as your old code)
            dz = torch.linspace(-1, 1, D, device=device)
            dy = torch.linspace(-1, 1, H, device=device)
            dx = torch.linspace(-1, 1, W, device=device)
            grid_z, grid_y, grid_x = torch.meshgrid(dz, dy, dx, indexing='ij')
            grid = torch.stack((grid_x, grid_y, grid_z), dim=3)[None, ...].repeat(B, 1, 1, 1, 1)

            flow = flow.permute(0, 2, 3, 4, 1)

            # Scale flow to [-1,1]
            flow_scaled = torch.zeros_like(flow)
            flow_scaled[..., 0] = 2.0 * flow[..., 0] / (W - 1)
            flow_scaled[..., 1] = 2.0 * flow[..., 1] / (H - 1)
            flow_scaled[..., 2] = 2.0 * flow[..., 2] / (D - 1)

            new_locs = grid + flow_scaled

        # Warp
        warped = F.grid_sample(src, new_locs, align_corners=False,
                               mode=mode, padding_mode='border')
        return warped

class LightResBlock(nn.Module):
    """
    Lightweight ResBlock with fewer parameters and memory usage.
    (FIXED to handle arbitrary in_channels for GroupNorm)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # --- THE FIX ---
        # Helper to find a valid group number (max 4)
        def find_groups(channels):
            if channels < 4:
                return 1
            # Try 4 groups first, then 2, then 1
            for g in [4, 2, 1]:
                if channels % g == 0:
                    return g
            return 1 # Fallback (1 always works)

        # groups for gn1 must be divisible by in_channels
        groups_gn1 = find_groups(in_channels)
        
        # Use 3x3 convs but with reduced intermediate channels
        mid_channels = max(in_channels // 2, out_channels // 2, 8)
        
        # groups for gn2 must be divisible by mid_channels
        groups_gn2 = find_groups(mid_channels)

        self.gn1 = nn.GroupNorm(groups_gn1, in_channels)
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
        self.gn2 = nn.GroupNorm(groups_gn2, mid_channels)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Shortcut connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x
        
        out = F.relu(self.gn1(x))
        out = self.conv1(out)
        out = F.relu(self.gn2(out))
        out = self.conv2(out)
        
        out = out + self.shortcut(residual)
        
        return out

class ResBlockSN(nn.Module):
    """Lightweight Spectral Norm ResBlock for Critic."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        self.conv2 = spectral_norm(nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride))

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.conv1(x), 0.2)
        out = self.conv2(out)
        out = out + self.shortcut(residual)
        out = F.leaky_relu(out, 0.2)
        return out

class ResBlockGN(nn.Module):
    """
    ResBlock with GroupNorm. (Replaces ResBlockSN)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Find valid group numbers
        def find_groups(channels):
            if channels < 4: return 1
            for g in [4, 2, 1]:
                if channels % g == 0: return g
            return 1

        groups_gn1 = find_groups(in_channels)
        mid_channels = out_channels // 2
        groups_gn2 = find_groups(mid_channels)

        self.gn1 = nn.GroupNorm(groups_gn1, in_channels)
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
        self.gn2 = nn.GroupNorm(groups_gn2, mid_channels)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # REMOVED spectral_norm
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x
        out = F.relu(self.gn1(x))
        out = self.conv1(out)
        out = F.relu(self.gn2(out))
        out = self.conv2(out)
        out = out + self.shortcut(residual)
        return out
    
# outputs low dimensional plan
class Planner(nn.Module):
    """UPDATED: Simpler planner for lower-dimensional features"""
    def __init__(self, in_channels=51, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Simpler encoder for 8x8x8 input
        self.encoder_conv1 = LightResBlock(in_channels, 16, stride=2)  # 8->4
        self.encoder_conv2 = LightResBlock(16, 32, stride=2)            # 4->2
        
        self.pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        
        # Input is now 32 * 2 * 2 * 2 = 256
        self.fc_mean = nn.Linear(32 * 2 * 2 * 2, latent_dim)
        self.fc_log_std = nn.Linear(32 * 2 * 2 * 2, latent_dim)
    
    def forward(self, state):
        x = self.encoder_conv1(state)
        x = self.encoder_conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, -4, 0)
        std = torch.exp(log_std)
        
        return torch.distributions.Normal(mean, std)
    
# deterministic decoder that turns plan into a flow field
class Actor(nn.Module):
    """UPDATED: Simpler actor for lower-dimensional latent space"""
    def __init__(self, latent_dim=32, feature_channels=8):
        super().__init__()
        
        # Project latent to 3D
        self.fc = nn.Linear(latent_dim, 32 * 2 * 2 * 2)
        
        # Simple decoder to 8x8x8
        self.decoder_conv1 = nn.Sequential(
            LightResBlock(32, 16, stride=1),
            nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2)  # 2->4
        )
        self.decoder_conv2 = nn.Sequential(
            LightResBlock(16, feature_channels, stride=1),
            nn.ConvTranspose3d(feature_channels, feature_channels, kernel_size=2, stride=2)  # 4->8
        )
        
        # Output flow at 8x8x8 resolution
        self.flow_head = nn.Conv3d(feature_channels, 3, kernel_size=3, padding=1)
    
    def forward(self, plan):
        x = self.fc(plan)
        x = x.view(x.size(0), 32, 2, 2, 2)
        x = self.decoder_conv1(x)
        x = self.decoder_conv2(x)
        flow = self.flow_head(x)
        
        # Output is 8x8x8 flow field
        return torch.tanh(flow) * 2.0  # Allow larger movements
    
# vlearn critic; only looks at state
class VCritic(nn.Module):
    """UPDATED: Simpler critic for lower-dimensional state"""
    def __init__(self, in_channels):
        super().__init__()
        
        # Simpler architecture for 8x8x8 input
        self.critic_conv1 = ResBlockGN(in_channels, 16, stride=2)  # 8->4
        self.critic_conv2 = ResBlockGN(16, 32, stride=2)           # 4->2
        
        self.pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        
        self.critic1 = nn.Linear(32 * 2 * 2 * 2, 1)
        self.critic2 = nn.Linear(32 * 2 * 2 * 2, 1)
    
    def forward(self, state):
        x = self.critic_conv1(state)
        x = self.critic_conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        val1 = self.critic1(x)
        val2 = self.critic2(x)
        
        # Output scaling for stability
        output_scale = 50.0  # Match expected reward magnitude
        final_val1 = torch.tanh(val1) * output_scale
        final_val2 = torch.tanh(val2) * output_scale
        
        return final_val1, final_val2
    
    def get_value(self, state):
        with torch.no_grad():
            val1, val2 = self.forward(state)
            return torch.min(val1, val2)

class VLearnPolicy(nn.Module):
    """
    Direct policy network (no separate planner/actor split).
    Paper shows this works better than Q-learning for high-dim actions.
    """
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        
        # Simple MLP for registration
        self.encoder = nn.Sequential(
            nn.Conv3d(state_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv3d(32, action_dim, 3, padding=1),
        )
        
        # Output mean and log_std for Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(1, action_dim, 1, 1, 1))
    
    def forward(self, state):
        features = self.encoder(state)
        mean = self.decoder(features)
        
        # Clamp log_std (paper uses [-2, 0.5])
        log_std = torch.clamp(self.log_std, -2.0, 0.5)
        std = log_std.exp()
        
        return torch.distributions.Normal(mean, std)


class VLearnCritic(nn.Module):
    """
    Twin V-critics (not Q-critics!).
    Paper uses this to reduce overestimation bias.
    """
    def __init__(self, state_dim, hidden_size=256):
        super().__init__()
        
        # Two independent V-networks
        self.v1 = self._build_v_network(state_dim, hidden_size)
        self.v2 = self._build_v_network(state_dim, hidden_size)
    
    def _build_v_network(self, state_dim, hidden_size):
        return nn.Sequential(
            nn.Conv3d(state_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, state):
        v1 = self.v1(state)
        v2 = self.v2(state)
        return v1, v2
    
    def get_value(self, state):
        """Return minimum of twin critics (paper recommendation)"""
        v1, v2 = self.forward(state)
        return torch.min(v1, v2)
    
class VLearnPolicy_Medium(nn.Module):
    def __init__(self, in_channels=128, use_checkpointing=True):
        super().__init__()
        print("--- ✅ Loading MEDIUM-CAPACITY VLearnPolicy ---")
        self.use_checkpointing = use_checkpointing
        self.adaptive_pool = nn.AdaptiveAvgPool3d((16, 16, 16)) 

        # --- Layer definitions (These are correct) ---
        self.encoder_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            LightResBlock(16, 16, stride=2) # 16 -> 8
        )
        self.encoder_conv2 = LightResBlock(16, 32, stride=2) # 8 -> 4
        self.encoder_conv3 = LightResBlock(32, 64, stride=2) # 4 -> 2
        
        self.decoder_conv1 = nn.Sequential(
            LightResBlock(64, 32, stride=1),
            nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2) # 2 -> 4
        )
        self.decoder_conv2 = nn.Sequential(
            LightResBlock(32 + 32, 16, stride=1), # +skip from encoder_conv2
            nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2) # 4 -> 8
        )
        self.decoder_conv3 = nn.Sequential(
            LightResBlock(16 + 16, 16, stride=1), # +skip from encoder_conv1
            nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2) # 8 -> 16
        )

        self.flow_mean = nn.Conv3d(16, 3, kernel_size=3, padding=1)
        self.flow_log_std = nn.Conv3d(16, 3, kernel_size=3, padding=1)

        self.critic_conv1 = ResBlockSN(in_channels, 16, stride=2) # 16 -> 8
        self.critic_conv2 = ResBlockSN(16, 32, stride=2)           # 8 -> 4
        self.critic_conv3 = ResBlockSN(32, 64, stride=2)           # 4 -> 2
        
        self.critic1 = spectral_norm(nn.Linear(64 * 2 * 2 * 2, 1))
        self.critic2 = spectral_norm(nn.Linear(64 * 2 * 2 * 2, 1))
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Conservative initialization for stability."""
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                # ... (rest of the conv initialization is fine) ...
                if 'flow' in name:
                    nn.init.xavier_normal_(m.weight, gain=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                # --- STABILITY FIX: Initialize critic heads to output near-zero values ---
                # A smaller std dev ensures the initial random outputs are small.
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Small initial std for exploration (this is for the actor, it's fine)
        if hasattr(self.flow_log_std, 'bias') and self.flow_log_std.bias is not None:
            nn.init.constant_(self.flow_log_std.bias, -1.0)

    def _run_block(self, block, x):
        """Helper for gradient checkpointing - ONLY when memory is really tight."""
        if self.use_checkpointing and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward_actor(self, state):
        state = self.adaptive_pool(state)
        
        # Encoder pass
        e1 = self._run_block(self.encoder_conv1, state)
        e2 = self._run_block(self.encoder_conv2, e1)
        e3 = self._run_block(self.encoder_conv3, e2) # This is the bottleneck

        # Decoder pass with correct skip connections
        d1 = self._run_block(self.decoder_conv1, e3)
        d2 = self._run_block(self.decoder_conv2, torch.cat([d1, e2], dim=1))
        d3 = self._run_block(self.decoder_conv3, torch.cat([d2, e1], dim=1))

        # Flow prediction
        flow_mean = self.flow_mean(d3)
        flow_log_std = self.flow_log_std(d3)

        return flow_mean, flow_log_std
    
    """
    def forward_critics(self, state):
        state = self.adaptive_pool(state)
        
        # Much more aggressive downsampling for critics to save memory
        x = self._run_block(self.critic_conv1, state)  # 16 x 32³
        x = self._run_block(self.critic_conv2, x)      # 32 x 8³
        x = self._run_block(self.critic_conv3, x)      # 64 x 4³
        
        # Global average pooling
        x = x.mean([2, 3, 4])
        
        return self.critic1(x), self.critic2(x)
    """
    
    def forward_critics(self, state):
        x = self.adaptive_pool(state)
        x = self._run_block(self.critic_conv1, x)
        x = self._run_block(self.critic_conv2, x)
        x = self._run_block(self.critic_conv3, x)
        
        # Flatten the output for the linear layer
        x = x.view(x.size(0), -1) 
        
        val1 = self.critic1(x)
        val2 = self.critic2(x)
        
        # The tanh squashing for stability is still a good idea
        output_scale = 10.0
        final_val1 = torch.tanh(val1) * output_scale
        final_val2 = torch.tanh(val2) * output_scale
        
        return final_val1, final_val2
    
    def get_action_dist(self, state):
        """Get action distribution with magnitude control."""
        raw_mean, log_std = self.forward_actor(state)
        
        # Get current max flow from the adaptive policy if it exists, otherwise use default
        if hasattr(self, 'get_current_max_flow'):
            max_flow_val = self.get_current_max_flow()
        else:
            max_flow_val = 10.0  # Default fallback
        
        # Soft clamping
        mean = max_flow_val * torch.tanh(raw_mean / max_flow_val)
        
        # Standard deviation bounds
        LOG_STD_MAX, LOG_STD_MIN = 0.5, -2.0  # Tighter bounds
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std).clamp(min=1e-6)
        
        # Debug information
        # if self.training:
        #     with torch.no_grad():
        #         mean_magnitude = torch.norm(mean, dim=1).mean()
        #         print(f"[DEBUG] Flow magnitude: {mean_magnitude:.2f} voxels")
        
        return torch.distributions.Normal(mean, std)
    
    def get_value(self, state):
        """
        Returns a single, deterministic value estimate for a given state.
        Uses the minimum of the twin critics to reduce overestimation.
        """
        # We don't need gradients when just getting the value for a target
        with torch.no_grad():
            val1, val2 = self.forward_critics(state)
            return torch.min(val1, val2)

class AdaptiveFlowPolicy_Medium(VLearnPolicy_Medium):
    def __init__(self, in_channels=128, use_checkpointing=True):
        super().__init__(in_channels, use_checkpointing)
        self.epoch = 0
        # More aggressive flow schedule for better exploration
        self.max_flow_schedule = {
            "phase1": 1.0,   # Epochs 0-75: More aggressive exploration
            "phase2": 0.5,    # Epochs 76-125: Moderate refinement phase 
            "phase3": 0.1     # Epochs 126+: Fine-tuning phase
        }
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def get_current_max_flow(self):
        """More aggressive flow scheduling for better exploration"""
        if self.epoch <= 75:
            return self.max_flow_schedule["phase1"] 
        elif self.epoch <= 125:
            return self.max_flow_schedule["phase2"] 
        else:
            return self.max_flow_schedule["phase3"]
    
    def get_action_dist(self, state):
        """Overrides the parent method to use a scheduled action space."""
        raw_mean, log_std = self.forward_actor(state)
        
        current_max_flow = self.get_current_max_flow()
        
        # Soft clamping of the action's mean to the current scheduled max
        mean = current_max_flow * torch.tanh(raw_mean / current_max_flow)
        
        # Simple std bounds - tighten over time
        if self.epoch <= 75:
            LOG_STD_MAX, LOG_STD_MIN = 0.0, -2.0
        else:
            LOG_STD_MAX, LOG_STD_MIN = -1.0, -2.5
            
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std).clamp(min=1e-6)
        
        return torch.distributions.Normal(mean, std)
    
class VLearnPolicy_RawImage(nn.Module):
    """
    An ablation version of the policy that takes raw images as input.
    The ONLY change is the number of input channels.
    """
    def __init__(self, in_channels=5, use_checkpointing=True): # MODIFIED: in_channels = 1+1+3 = 5
        super().__init__()
        print(f"--- ✅ Loading RAW IMAGE VLearnPolicy (in_channels={in_channels}) ---")

        self.use_checkpointing = use_checkpointing
        self.adaptive_pool = nn.AdaptiveAvgPool3d((16, 16, 16))

        # --- MUCH SMALLER ENCODER ---
        # Reduce initial channels and use less aggressive downsampling
        self.encoder_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=5, stride=2, padding=2), 
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
            LightResBlock(8, 8, stride=1)  # No downsampling here
        )
        
        self.encoder_conv2 = nn.Sequential(
            LightResBlock(8, 16, stride=2),  # 64->32
        )
        
        self.encoder_conv3 = nn.Sequential(
            LightResBlock(16, 32, stride=2),  # 32->16
        )
        
        self.encoder_conv4 = nn.Sequential(
            LightResBlock(32, 64, stride=2),  # 16->8
        )
        
        # Skip the very deep layers to save memory
        # self.encoder_conv5 and bottleneck removed
        
        # --- LIGHTWEIGHT DECODER ---
        # Start from 64 channels instead of 512
        self.decoder_conv1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),  # 8->16
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            LightResBlock(32, 32, stride=1)
        )
        
        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose3d(32 + 32, 16, kernel_size=2, stride=2),  # 16->32, +skip from encoder_conv3
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            LightResBlock(16, 16, stride=1)
        )
        
        self.decoder_conv3 = nn.Sequential(
            nn.ConvTranspose3d(16 + 16, 8, kernel_size=2, stride=2),   # 32->64, +skip from encoder_conv2
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
            LightResBlock(8, 8, stride=1)
        )
        
        self.decoder_conv4 = nn.Sequential(
            nn.ConvTranspose3d(8 + 8, 8, kernel_size=2, stride=2),    # 64->128, +skip from encoder_conv1
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
            LightResBlock(8, 8, stride=1)
        )

        # --- SIMPLE FLOW OUTPUT HEADS ---
        self.flow_mean = nn.Conv3d(8, 3, kernel_size=1)  # Direct prediction, no intermediate layers
        self.flow_log_std = nn.Conv3d(8, 3, kernel_size=1)

        # --- LIGHTWEIGHT CRITICS ---
        # Much smaller critic network
        self.critic_conv1 = ResBlockSN(in_channels, 16, stride=4) # Aggressive downsampling: 128->32
        self.critic_conv2 = ResBlockSN(16, 32, stride=4)           # 32->8  
        self.critic_conv3 = ResBlockSN(32, 64, stride=2)           # 8->4
        
        self.critic1 = spectral_norm(nn.Linear(64, 1))
        self.critic2 = spectral_norm(nn.Linear(64, 1))

        self._initialize_weights()

    def _initialize_weights(self):
        """Conservative initialization for stability."""
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                # ... (rest of the conv initialization is fine) ...
                if 'flow' in name:
                    nn.init.xavier_normal_(m.weight, gain=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                # --- STABILITY FIX: Initialize critic heads to output near-zero values ---
                # A smaller std dev ensures the initial random outputs are small.
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Small initial std for exploration (this is for the actor, it's fine)
        if hasattr(self.flow_log_std, 'bias') and self.flow_log_std.bias is not None:
            nn.init.constant_(self.flow_log_std.bias, -1.0)

    def _run_block(self, block, x):
        """Helper for gradient checkpointing - ONLY when memory is really tight."""
        if self.use_checkpointing and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward_actor(self, state):
        state = self.adaptive_pool(state)
        
        # --- ENCODER (with optional checkpointing) ---
        e1 = self._run_block(self.encoder_conv1, state)  # 8 x 64³
        e2 = self._run_block(self.encoder_conv2, e1)     # 16 x 32³
        e3 = self._run_block(self.encoder_conv3, e2)     # 32 x 16³
        e4 = self._run_block(self.encoder_conv4, e3)     # 64 x 8³

        # --- DECODER with Skip Connections ---
        d1 = self._run_block(self.decoder_conv1, e4)                            # 32 x 16³
        d2 = self._run_block(self.decoder_conv2, torch.cat([d1, e3], dim=1))    # 16 x 32³
        d3 = self._run_block(self.decoder_conv3, torch.cat([d2, e2], dim=1))    # 8 x 64³
        d4 = self._run_block(self.decoder_conv4, torch.cat([d3, e1], dim=1))    # 8 x 128³

        # Flow prediction
        flow_mean = self.flow_mean(d4)
        flow_log_std = self.flow_log_std(d4)

        return flow_mean, flow_log_std
    
    """
    def forward_critics(self, state):
        state = self.adaptive_pool(state)
        
        # Much more aggressive downsampling for critics to save memory
        x = self._run_block(self.critic_conv1, state)  # 16 x 32³
        x = self._run_block(self.critic_conv2, x)      # 32 x 8³
        x = self._run_block(self.critic_conv3, x)      # 64 x 4³
        
        # Global average pooling
        x = x.mean([2, 3, 4])
        
        return self.critic1(x), self.critic2(x)
    """
    
    def forward_critics(self, state):
        x = self.adaptive_pool(state)
        x = self._run_block(self.critic_conv1, x)
        x = self._run_block(self.critic_conv2, x)
        x = self._run_block(self.critic_conv3, x)
        x = x.mean([2, 3, 4])
        
        val1 = self.critic1(x)
        val2 = self.critic2(x)
        
        # --- FINAL STABILITY FIX: Squash and scale the final output ---
        # tanh squashes the output of the linear layers to a [-1, 1] range.
        # We then scale it to a reasonable range for a value function.
        # This completely prevents the critic from ever outputting enormous values.
        output_scale = 10.0 # A hyperparameter, but 10.0 is a robust start
        
        final_val1 = torch.tanh(val1) * output_scale
        final_val2 = torch.tanh(val2) * output_scale
        
        return final_val1, final_val2

    def get_action_dist(self, state):
        """Get action distribution with magnitude control."""
        raw_mean, log_std = self.forward_actor(state)
        
        # Get current max flow from the adaptive policy if it exists, otherwise use default
        if hasattr(self, 'get_current_max_flow'):
            max_flow_val = self.get_current_max_flow()
        else:
            max_flow_val = 10.0  # Default fallback
        
        # Soft clamping
        mean = max_flow_val * torch.tanh(raw_mean / max_flow_val)
        
        # Standard deviation bounds
        LOG_STD_MAX, LOG_STD_MIN = 0.5, -2.0  # Tighter bounds
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std).clamp(min=1e-6)
        
        return torch.distributions.Normal(mean, std)
    
    def get_value(self, state):
        """
        Returns a single, deterministic value estimate for a given state.
        Uses the minimum of the twin critics to reduce overestimation.
        """
        # We don't need gradients when just getting the value for a target
        with torch.no_grad():
            val1, val2 = self.forward_critics(state)
            return torch.min(val1, val2)

class AdaptiveFlowPolicy_Raw(VLearnPolicy_RawImage): # Inherits from the new raw policy
    def __init__(self, in_channels=5, use_checkpointing=True):
        super().__init__(in_channels, use_checkpointing)
        self.epoch = 0
        # More conservative and gradual schedule for raw images
        self.max_flow_schedule = {
            "phase1": 3.0,   # Start lower for raw images
            "phase2": 1.5,   # More conservative refinement
            "phase3": 0.3    # Very fine tuning
        }
        
    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_current_max_flow(self):
        """Simple phase-based flow scheduling for raw images"""
        if self.epoch <= 75:
            return self.max_flow_schedule["phase1"]
        elif self.epoch <= 125:
            return self.max_flow_schedule["phase2"] 
        else:
            return self.max_flow_schedule["phase3"]

    def get_action_dist(self, state):
        """Overrides the parent method to use a scheduled action space."""
        raw_mean, log_std = self.forward_actor(state)
        
        current_max_flow = self.get_current_max_flow()
        
        # Simple std bounds - tighten over time for raw images
        if self.epoch <= 75:
            LOG_STD_MAX, LOG_STD_MIN = -0.3, -1.7
        else:
            LOG_STD_MAX, LOG_STD_MIN = -0.8, -2.3
            
        # Soft clamping of the action's mean
        mean = current_max_flow * torch.tanh(raw_mean / current_max_flow)
        
        # Apply log std bounds
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std).clamp(min=1e-6)
        
        return torch.distributions.Normal(mean, std)

# Patch for SpatialTransformer to ensure src always has 5 dims
def patched_forward(self, src, flow=None, grid=None, mode='bilinear'):
    """
    Enhanced forward with proper dimension handling and debugging.
    """
    # Debug print to help identify the issue
    if src.ndim >= 6:
        print(f"DEBUG: Unexpected tensor dimensions: {src.shape}")
        
    if src.ndim == 3:  # [D, H, W] -> [1, 1, D, H, W]
        src = src.unsqueeze(0).unsqueeze(0)
    elif src.ndim == 4:  # [B, D, H, W] or [C, D, H, W] -> [B, 1, D, H, W] or [1, C, D, H, W]
        if src.shape[0] <= 4:  # Likely [B, D, H, W] where B is small
            src = src.unsqueeze(1)
        else:  # Likely [C, D, H, W] where C is large (like 64 for features)
            src = src.unsqueeze(0)
    elif src.ndim == 5:  # Already correct: [B, C, D, H, W]
        pass
    elif src.ndim >= 6:  # Handle unexpected high-dimensional tensors
        # Try to squeeze out singleton dimensions
        while src.ndim > 5 and src.shape[0] == 1:
            src = src.squeeze(0)
        # If still too many dims, take the first element
        if src.ndim > 5:
            print(f"WARNING: Had to slice tensor from {src.shape} to handle excessive dimensions")
            src = src[0]  # Take first element along first dimension
        # Ensure it's now in a valid format
        if src.ndim == 4:
            if src.shape[0] <= 4:
                src = src.unsqueeze(1)
            else:
                src = src.unsqueeze(0)
        elif src.ndim == 3:
            src = src.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected src dimension: {src.ndim}. Expected 3, 4, or 5.")
    
    # Now proceed with the original forward logic
    B, C, D, H, W = src.shape
    device = src.device

    if flow is not None and (D, H, W) != flow.shape[2:]:
        flow = F.interpolate(flow, size=(D, H, W), mode='trilinear', align_corners=False)
    
    # If precomputed grid is given, just use it
    if grid is not None:
        new_locs = grid
    else:
        # Otherwise build it (same as your old code)
        dz = torch.linspace(-1, 1, D, device=device)
        dy = torch.linspace(-1, 1, H, device=device)
        dx = torch.linspace(-1, 1, W, device=device)
        grid_z, grid_y, grid_x = torch.meshgrid(dz, dy, dx, indexing='ij')
        grid = torch.stack((grid_x, grid_y, grid_z), dim=3)[None, ...].repeat(B, 1, 1, 1, 1)

        flow = flow.permute(0, 2, 3, 4, 1)

        # Scale flow to [-1,1]
        flow_scaled = torch.zeros_like(flow)
        flow_scaled[..., 0] = 2.0 * flow[..., 0] / (W - 1)
        flow_scaled[..., 1] = 2.0 * flow[..., 1] / (H - 1)
        flow_scaled[..., 2] = 2.0 * flow[..., 2] / (D - 1)

        new_locs = grid + flow_scaled

    # Warp
    warped = F.grid_sample(src, new_locs, align_corners=False,
                           mode=mode, padding_mode='border')
    return warped

SpatialTransformer.forward = patched_forward