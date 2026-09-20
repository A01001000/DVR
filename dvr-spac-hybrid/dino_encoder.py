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
        self._device = device
        self.patch_size = patch_size
        self.embed_dim = 1024
        self.resize = Resize((224, 224), antialias=True) 
        self.dino = self.load_dino()
        self.dino.eval()
        # Initialize with default single-layer dimension, will be updated if needed
        self.proj_head = ProjectionHead(self.embed_dim, 256).to(self._device)
    
    @property
    def device(self):
        """Return the device of the DINO model"""
        try:
            return next(self.parameters()).device
        except StopIteration:
            # No parameters yet, return the initial device
            return self._device

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
        state_dict = torch.load(model_path, map_location=self._device)
        
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
        return dino.to(self._device)


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

    def encode_volume(self, volume_3d, use_proj=True, dataset_type=None):
        """
        Enhanced volume encoding with brain-specific multi-layer fusion
        Args:
            volume_3d (tensor): [D, H, W]
            dataset_type (str): "brain" or "abdomen" to select appropriate strategy
        Returns:
            Flattened feature tensor: [D * N_patches, feat_dim]
        """
        if volume_3d.dim() != 3:
            raise ValueError(f"encode_volume expects a 3D tensor [D, H, W], but got {volume_3D.dim()}D")

        # Add a channel dim to treat slices as a batch: [D, 1, H, W]
        slice_batch = volume_3d.unsqueeze(1)
        
        if dataset_type and dataset_type.lower() == "brain":
            # For brain: Use multi-layer fusion for better structural understanding
            print("Using multi-layer DINO fusion for brain dataset")
            feats_batched = self.extract_brain_multilayer_features(slice_batch.to(self.device))
        else:
            # For abdomen: Use single layer extraction
            layer_idx = 9 if dataset_type and dataset_type.lower() == "abdomen" else None
            if layer_idx:
                print(f"Using layer {layer_idx} for abdomen images")
            feats_batched = self.extract_layer_specific_features(
                slice_batch.to(self.device), layer_idx=layer_idx
            )
        
        # Reshape to flattened features
        feats_flat = feats_batched.reshape(-1, feats_batched.shape[-1])
        
        # Apply projection head if needed
        return self.proj_head(feats_flat) if use_proj else feats_flat
    
    def extract_layer_specific_features(self, slice_batched_tensor, layer_idx=None):
        """
        Extracts features from a specific intermediate layer based on dataset type
        
        Args:
            slice_batched_tensor (tensor): [B, C, H, W] batch of grayscale image slices
            layer_idx (int): Which transformer layer to extract from (None=use final layer)
            
        Returns:
            tensor of shape [B, N_patches, feat_dim]
        """
        # Resize to DINO's expected input size -> [B, 1, 224, 224]
        slice_resized = self.resize(slice_batched_tensor)
        
        # Repeat the channel dimension to create a 3-channel image -> [B, 3, 224, 224]
        img3c = slice_resized.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            if layer_idx is not None:
                # Extract from specific intermediate layer using the correct DINO method
                # get_intermediate_layers expects n as either int (last n layers) or sequence of layer indices
                # Since we want a specific layer, we pass it as a list
                intermediate_outputs = self.dino.get_intermediate_layers(
                    img3c, n=[layer_idx], return_class_token=False, norm=True
                )
                # intermediate_outputs is a tuple, take the first (and only) element
                features = intermediate_outputs[0]  # Shape: [B, N_patches, feat_dim]
                return features
            else:
                # Use standard final layer extraction
                features_dict = self.dino.forward_features(img3c)
                return features_dict['x_norm_patchtokens']  # Shape: [B, N_patches, feat_dim]

    def extract_brain_multilayer_features(self, slice_batched_tensor):
        """
        Extract and fuse features from multiple DINO layers for better brain structure understanding.
        Uses layers that capture different levels of anatomical detail.
        
        Args:
            slice_batched_tensor (tensor): [B, C, H, W] batch of grayscale image slices
            
        Returns:
            tensor of shape [B, N_patches, feat_dim]
        """
        # Resize to DINO's expected input size -> [B, 1, 224, 224]
        slice_resized = self.resize(slice_batched_tensor)
        
        # Repeat the channel dimension to create a 3-channel image -> [B, 3, 224, 224]
        img3c = slice_resized.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            # Extract features from multiple layers
            # Layer 3: Early spatial features (good for boundaries)
            # Layer 6: Mid-level features (good for structures)  
            # Layer 9: Higher-level features (good for semantics)
            layers_to_extract = [3, 6, 9]
            
            intermediate_outputs = self.dino.get_intermediate_layers(
                img3c, n=layers_to_extract, return_class_token=False, norm=True
            )
            
            # intermediate_outputs is a tuple of features from each layer
            layer_features = list(intermediate_outputs)
            
            # FIXED: Ensure we have valid features
            for i, feat in enumerate(layer_features):
                if feat.shape[-1] == 0:
                    print(f"ERROR: Layer {layers_to_extract[i]} has zero feature dimension")
                    # Fallback to final layer extraction
                    features_dict = self.dino.forward_features(img3c)
                    return features_dict['x_norm_patchtokens']
            
            # Fuse features with learned attention weights
            # Different layers contribute differently to brain registration
            weights = [0.4, 0.35, 0.25]  # Early layers get more weight for spatial detail
            
            # Simple weighted sum fusion
            fused_features = weights[0] * layer_features[0]
            for i in range(1, len(weights)):
                fused_features += weights[i] * layer_features[i]
            
            return fused_features  # Shape: [B, N_patches, feat_dim]

    def freeze_dino(self): # since only training projection head
        for p in self.dino.parameters():
            p.requires_grad = False

    def train_proj_head(self, dataset, save_path, epochs=50, dataset_type=None):
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
                
                # Use brain-specific contrastive loss for better anatomical learning
                if dataset_type and dataset_type.lower() == "brain":
                    loss = self.brain_anatomical_contrastive_loss(f_mr, f_ct, temperature=0.05)
                else:
                    loss = self.contrastive_loss(f_mr, f_ct, temperature=0.07)

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
    
    @staticmethod
    def spatial_contrastive_loss(f_mr, f_ct, spatial_weight=0.1, temperature=0.07):
        """
        Enhanced contrastive loss that considers spatial proximity for medical images.
        This helps the model learn better correspondences for brain registration.
        
        Args:
            f_mr (tensor): [N, feat_dim] MR features
            f_ct (tensor): [N, feat_dim] CT features  
            spatial_weight (float): Weight for spatial proximity component
            temperature (float): Temperature for contrastive learning
        """
        # Standard contrastive loss
        standard_loss = DINOEncoder.contrastive_loss(f_mr, f_ct, temperature)
        
        # Add spatial awareness for better brain registration
        N = f_mr.size(0)
        
        # Create spatial proximity matrix (assumes features are ordered spatially)
        # For brain, nearby patches should have similar features
        spatial_indices = torch.arange(N).to(f_mr.device)
        spatial_distances = torch.abs(spatial_indices.unsqueeze(0) - spatial_indices.unsqueeze(1))
        spatial_weights = torch.exp(-spatial_distances / (N * 0.1))  # Decay with distance
        
        # Normalize features
        f_mr_norm = F.normalize(f_mr, dim=-1)
        f_ct_norm = F.normalize(f_ct, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(f_mr_norm, f_ct_norm.T) / temperature
        
        # Apply spatial weighting to encourage nearby patches to be similar
        spatial_loss = -torch.sum(spatial_weights * torch.log_softmax(sim_matrix, dim=1)) / N
        
        return standard_loss + spatial_weight * spatial_loss
    
    @staticmethod
    def brain_anatomical_contrastive_loss(f_mr, f_ct, temperature=0.05):
        """
        Brain-specific contrastive loss that encourages learning anatomical correspondences
        between MR and CT features by using harder negatives and anatomical structure awareness.
        """
        # Normalize features
        f_mr_norm = F.normalize(f_mr, dim=-1)
        f_ct_norm = F.normalize(f_ct, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(f_mr_norm, f_ct_norm.T) / temperature
        labels = torch.arange(f_mr.size(0), device=f_mr.device)
        
        # Standard contrastive loss
        loss_mr_to_ct = F.cross_entropy(sim_matrix, labels)
        loss_ct_to_mr = F.cross_entropy(sim_matrix.T, labels)
        
        # Add hard negative mining for brain structures
        # Focus on patches that are similar but shouldn't be (e.g., different brain regions)
        with torch.no_grad():
            # Find hard negatives (high similarity but wrong correspondence)
            mask = torch.eye(f_mr.size(0), device=f_mr.device).bool()
            sim_matrix_masked = sim_matrix.masked_fill(mask, float('-inf'))
            hard_negatives = sim_matrix_masked.max(dim=1)[0]
            
            # Weight loss more heavily for samples with strong hard negatives
            hard_negative_weights = torch.sigmoid(hard_negatives * 2)  # Sigmoid to keep weights reasonable
        
        # Apply hard negative weighting
        weighted_loss_mr_to_ct = (loss_mr_to_ct * hard_negative_weights).mean()
        weighted_loss_ct_to_mr = (loss_ct_to_mr * hard_negative_weights).mean()
        
        return (weighted_loss_mr_to_ct + weighted_loss_ct_to_mr) / 2.0

    @staticmethod
    def brain_anatomical_contrastive_loss(f_mr, f_ct, temperature=0.05):
        """
        Brain-specific contrastive loss that encourages learning anatomical correspondences
        between MR and CT features by using harder negatives and anatomical structure awareness.
        """
        # Normalize features
        f_mr_norm = F.normalize(f_mr, dim=-1)
        f_ct_norm = F.normalize(f_ct, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(f_mr_norm, f_ct_norm.T) / temperature
        labels = torch.arange(f_mr.size(0), device=f_mr.device)
        
        # Standard contrastive loss
        loss_mr_to_ct = F.cross_entropy(sim_matrix, labels)
        loss_ct_to_mr = F.cross_entropy(sim_matrix.T, labels)
        
        # Add hard negative mining for brain structures
        # Focus on patches that are similar but shouldn't be (e.g., different brain regions)
        with torch.no_grad():
            # Find hard negatives (high similarity but wrong correspondence)
            mask = torch.eye(f_mr.size(0), device=f_mr.device).bool()
            sim_matrix_masked = sim_matrix.masked_fill(mask, float('-inf'))
            hard_negatives = sim_matrix_masked.max(dim=1)[0]
            
            # Weight loss more heavily for samples with strong hard negatives
            hard_negative_weights = torch.sigmoid(hard_negatives * 2)  # Sigmoid to keep weights reasonable
        
        # Apply hard negative weighting
        weighted_loss_mr_to_ct = (loss_mr_to_ct * hard_negative_weights).mean()
        weighted_loss_ct_to_mr = (loss_ct_to_mr * hard_negative_weights).mean()
        
        return (weighted_loss_mr_to_ct + weighted_loss_ct_to_mr) / 2.0

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
        state = torch.load(path, map_location=self._device)
        self.proj_head.load_state_dict(state['proj_head'])
