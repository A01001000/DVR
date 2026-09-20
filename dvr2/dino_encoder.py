import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from tqdm import tqdm

from dvr2.dinov2.vision_transformer import vit_large
from dvr2.utils import get_augmentation_transform
from dvr2.dataset import HeadDataset


class ProjectionHead(nn.Module):
    """
    Transform input feature vectors into a lower-dimensional, normalized feature space (for contrastive learning)
    """
    def __init__(self, in_dim=1024, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


class DINOEncoder(nn.Module):
    def __init__(self, device='cuda', patch_size=14):
        super().__init__()
        self.device = device
        self.patch_size = patch_size
        self.embed_dim = 1024
        self.resize = Resize((224, 224), antialias=True) 
        self.dino = self.load_dino()
        self.dino.eval()
        self.proj_head = ProjectionHead(self.embed_dim, 256).to(self.device)

    def load_dino(self):
        model_path = './dvr2/dinov2/dinov2_vitl14_reg4_pretrain.pth'
    
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"DINO model checkpoint not found at '{model_path}'. "
                f"Please download it manually from:\n"
                f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
            )

        # This creates the 'dinov2_vitl14' architecture
        dino = vit_large(patch_size=14, init_values=1.0, block_chunks=0, num_register_tokens=4)
        state_dict = torch.load(model_path, map_location=self.device)
        
        pos_embed_checkpoint = state_dict['pos_embed']
        pos_embed_model = dino.pos_embed
        
        # Check if they don't match
        if pos_embed_checkpoint.shape != pos_embed_model.shape:
            print(f"Resizing positional embedding from {pos_embed_checkpoint.shape} to {pos_embed_model.shape}")
            
            # Separate the class token and patch tokens
            cls_token_checkpoint = pos_embed_checkpoint[:, 0:1, :]
            patch_embed_checkpoint = pos_embed_checkpoint[:, 1:, :] # Shape [1, N_patches_old, C]
            
            # Get the number of patches in the new model (e.g., 256 for a 224x224 image)
            num_patches_model = pos_embed_model.shape[1] - 1
            
            # Calculate the old and new grid sizes (assuming square)
            gs_old = int(patch_embed_checkpoint.shape[1]**0.5)
            gs_new = int(num_patches_model**0.5)
            
            # Reshape to a 2D grid and interpolate
            # Shape: [1, C, gs_old, gs_old]
            patch_embed_resized = patch_embed_checkpoint.permute(0, 2, 1).reshape(1, -1, gs_old, gs_old)
            # Shape: [1, C, gs_new, gs_new]
            patch_embed_resized = F.interpolate(patch_embed_resized, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
            
            # Reshape back to the sequence format
            # Shape: [1, N_patches_new, C]
            patch_embed_resized = patch_embed_resized.reshape(1, -1, gs_new * gs_new).permute(0, 2, 1)
            
            # Concatenate the class token back
            new_pos_embed = torch.cat((cls_token_checkpoint, patch_embed_resized), dim=1)
            
            # Update the state dictionary with the resized embedding
            state_dict['pos_embed'] = new_pos_embed
            
        dino.load_state_dict(state_dict, strict=True)
        dino.head = nn.Identity()
        return dino.to(self.device)


    def extract_dino_features(self, slice_batched_tensor):
        """
        Extracts patch-level features

        Args:
            slice_batched_tensor (tensor): [B, C, H, W] batch of grayscale image slices.

        Returns:
            tensor of shape [N_patches, 1024]
        """

        # Resize to DINO's expected input size -> [B, 1, 224, 224]
        slice_resized = self.resize(slice_batched_tensor) # -> [B, 1, 224, 224]

        # Repeat the channel dimension to create a 3-channel image -> [1, B, 3, 224, 224]
        img3c = slice_resized.repeat(1, 3, 1, 1) # -> [B, 3, 224, 224]
    
        # Now the tensor has the correct [B, C, H, W] shape for DINO
        with torch.no_grad():
            # The DINO model will now handle the positional encoding interpolation itself
            features_dict = self.dino.forward_features(img3c)
            feats = features_dict['x_norm_patchtokens'] # Shape: [B, N_patches, 1024]
        
        return feats # Shape: [B * N_patches, 1024]

    def forward(self, slice_tensor, use_proj=True):
        """
        Accepts a 4D batch [B, C, H, W] and returns FLATTENED features for all patches.
        """
        feats_batched = self.extract_dino_features(slice_tensor.to(self.device)) # -> [B, N_patches, 1024]
        feats_flat = feats_batched.reshape(-1, self.embed_dim) # -> [B * N_patches, 1024]
        return self.proj_head(feats_flat) if use_proj else feats_flat

    def encode_volume(self, volume_3d, use_proj=True):
        """
        CORRECTED: Efficiently processes a 3D volume and returns the flattened features.
        Args:
            volume_3d (tensor): [D, H, W]
        Returns:
            Flattened feature tensor: [D * N_patches, feat_dim]
        """
        if volume_3d.dim() != 3:
            raise ValueError(f"encode_volume expects a 3D tensor [D, H, W], but got {volume_3d.dim()}D")

        # Add a channel dim to treat slices as a batch: [D, 1, H, W]
        slice_batch = volume_3d.unsqueeze(1)
        
        # Pass the entire batch of slices. The `forward` method returns the
        # correctly flattened features for all slices in the volume.
        return self.forward(slice_batch, use_proj=use_proj)

    def freeze_dino(self): # since only training projection head
        for p in self.dino.parameters():
            p.requires_grad = False

    def train_proj_head(self, dataset, save_path, epochs=50):
        optimizer = torch.optim.Adam(self.proj_head.parameters(), lr=1e-4)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

        for epoch in range(epochs):
            pbar = tqdm(loader)
            for mr_slice, ct_slice in pbar:
                mr_slice = mr_slice.to(self.device)
                ct_slice = ct_slice.to(self.device)

                # self.forward already returns the features flattened and projected
                f_mr = self.forward(mr_slice, use_proj=True)
                f_ct = self.forward(ct_slice, use_proj=True)
                
                # The .view() calls that were here are no longer needed.
                loss = self.contrastive_loss(f_mr, f_ct)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
                
        if save_path:
            self.save(save_path)
            print(f"Projection head saved to {save_path}")

    @staticmethod
    def contrastive_loss(f_mr, f_ct, temperature=0.07):
        """
        Computes a contrastive loss between two sets of features (e.g., MRI and CT) using cross-entropy on their similarity scores.
        Specifically, it gets InfoNCE (Noise Contrastive Estimation)

        Args:
            f_mr (tensor): [N, feat_dim]
            f_ct (tensor): [N, feat_dim]

        Returns:
            scalar tensor
        """
        # Normalize features to prevent numerical instability and scale logits
        f_mr = F.normalize(f_mr, dim=-1)
        f_ct = F.normalize(f_ct, dim=-1)
    
        # Calculate similarity
        logits = torch.matmul(f_mr, f_ct.T) / temperature
        labels = torch.arange(len(f_mr)).to(f_mr.device)

        # Calculate loss in both directions
        loss_mr_to_ct = F.cross_entropy(logits, labels)
        loss_ct_to_mr = F.cross_entropy(logits.T, labels) # Note the transpose on logits

        # Return the average
        return (loss_mr_to_ct + loss_ct_to_mr) / 2.0

    def save(self, path):
        """
        Saves state of the projection head to a file
        """
        torch.save({
            'proj_head': self.proj_head.state_dict()
        }, path)

    def load(self, path):
        """
        Loads the projection head state from a file.
        """
        state = torch.load(path, map_location=self.device)
        self.proj_head.load_state_dict(state['proj_head'])

