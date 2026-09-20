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
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Use smaller groups for GroupNorm
        groups = min(4, out_channels // 4) if out_channels >= 4 else 1
        
        # Use 3x3 convs but with reduced intermediate channels
        mid_channels = max(in_channels // 2, out_channels // 2, 8)
        
        self.gn1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
        self.gn2 = nn.GroupNorm(min(4, mid_channels // 4) if mid_channels >= 4 else 1, mid_channels)
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


class VLearnPolicy_Medium(nn.Module):
    def __init__(self, in_channels=131, use_checkpointing=True):
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
        
        # Soft clamping
        mean = self.max_flow_val * torch.tanh(raw_mean / self.max_flow_val)
        
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

class AdaptiveFlowPolicy_Medium(VLearnPolicy_Medium): # Inherits from the new medium policy
    def __init__(self, in_channels=131, use_checkpointing=True):
        super().__init__(in_channels, use_checkpointing)
        self.epoch = 0
        self.max_flow_schedule = {
            "phase1": 5.0, "phase2": 2.0, "phase3": 0.5
        }
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def get_current_max_flow(self):
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
        
        # Also tighten the standard deviation as we refine
        if self.epoch <= 75:
            LOG_STD_MAX, LOG_STD_MIN = 0.0, -2.0
        else:
            LOG_STD_MAX, LOG_STD_MIN = -1.0, -2.5
            
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std).clamp(min=1e-6)
        
        return torch.distributions.Normal(mean, std)