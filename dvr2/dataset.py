import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import torchio as tio
    
class DistillationDataset(Dataset):
    def __init__(self, data_dir):
        """
        This dataset simply loads the pre-downsampled images
        and pre-computed features from the disk.
        """
        self.image_dir = os.path.join(data_dir, "images")
        self.feature_dir = os.path.join(data_dir, "features")
        self.sample_names = sorted([f.replace('_image.pt', '') for f in os.listdir(self.image_dir)])

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        sample_name = self.sample_names[idx]
        image_path = os.path.join(self.image_dir, f"{sample_name}_image.pt")
        feature_path = os.path.join(self.feature_dir, f"{sample_name}_features.pt")

        # Load the pre-downsampled image and target features
        # The saved image tensor is 3D: [D, H, W]
        input_image = torch.load(image_path)
        target_features = torch.load(feature_path)
        
        # Add the channel dimension to the image, making it 4D: [1, D, H, W]
        # The F.interpolate call has been removed.
        return input_image.unsqueeze(0), target_features
    
class TestDataset(Dataset):
    """
    A PyTorch Dataset that wraps pre-loaded data lists and
    loads corresponding cached features on the fly.
    """
    def __init__(self, vol_pairs, label_pairs, target_size=(128, 128, 128)):
        # Store the lists of pre-loaded volumes and labels
        self.vol_pairs = vol_pairs
        self.label_pairs = label_pairs
        self.target_size = target_size

        assert len(self.vol_pairs) == len(self.label_pairs), "Mismatch between volumes and labels"

    def __len__(self):
        # The total number of test samples
        return len(self.vol_pairs)

    def __getitem__(self, idx):
        # Get the pre-loaded volumes and labels
        mr_path, ct_path = self.vol_pairs[idx] # Shape: [D, H, W]
        mr_label_path, ct_label_path = self.label_pairs[idx] # Shape: [D, H, W]
        
        subject = tio.Subject(
            mr=tio.ScalarImage(mr_path),
            ct=tio.ScalarImage(ct_path), 
            mr_seg=tio.LabelMap(mr_label_path),
            ct_seg=tio.LabelMap(ct_label_path)
        )

        # --- PREPROCESSING FIX: Align the MR to the CT's physical space ---
        # Create a transform that will resample any image onto the CT's grid
        resampler = tio.Resample(target=subject.ct)
        # Apply it to the MR image
        mr_aligned = resampler(subject.mr)
        mr_label_aligned = resampler(subject.mr_seg)

        # Create a new subject with the now-aligned MR and original CT
        aligned_subject = tio.Subject(
            mr=mr_aligned,
            ct=subject.ct,
            mr_seg=mr_label_aligned,
            ct_seg=subject.ct_seg
        )

        mr_vol = aligned_subject.mr.data
        ct_vol = aligned_subject.ct.data
        mr_seg = aligned_subject.mr_seg.data
        ct_seg = aligned_subject.ct_seg.data

        # --- MODIFIED & ROBUST DOWNSAMPLING ---
        # This function handles both 3D and 4D inputs safely
        def robust_interpolate(tensor_in, target_size, mode):
            # If the input is already 4D [C, D, H, W], just add a batch dim.
            # This is the case for your NEW model.
            if tensor_in.dim() == 4:
                tensor_5d = tensor_in.unsqueeze(0)
            # If the input is 3D [D, H, W], add batch and channel dims.
            # This is the case for your OLD models.
            elif tensor_in.dim() == 3:
                tensor_5d = tensor_in.unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError(f"Unsupported tensor dimension: {tensor_in.dim()}")

            # Use align_corners=False for trilinear, but it's not supported for nearest
            interp_kwargs = {'align_corners': False} if mode != 'nearest' else {}
            return F.interpolate(tensor_5d, size=target_size, mode=mode, **interp_kwargs).squeeze(0)

        # Downsample images using the robust function
        mr_img_downsampled = robust_interpolate(mr_vol, self.target_size, mode='trilinear')
        ct_img_downsampled = robust_interpolate(ct_vol, self.target_size, mode='trilinear')
        
        # Downsample labels (note the .float() conversion for interpolation)
        mr_seg_downsampled = robust_interpolate(mr_seg.float(), self.target_size, mode='nearest').long()
        ct_seg_downsampled = robust_interpolate(ct_seg.float(), self.target_size, mode='nearest').long()
        
        final_spacing = aligned_subject.ct.spacing
        
        return mr_img_downsampled, ct_img_downsampled, final_spacing, mr_seg_downsampled, ct_seg_downsampled

class HeadDataset(Dataset):
    def __init__(self, path_pairs, transform=None, target_size=(128, 128, 128)):
        """
        path_pairs: A list of (mr_path, ct_path) tuples.
        transform: A TorchIO transform pipeline to be applied on the fly.
        """
        self.path_pairs = path_pairs
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.path_pairs)

    def __getitem__(self, idx):
        mr_path, ct_path = self.path_pairs[idx]
        
        # Create a TorchIO Subject for consistent transformation
        subject = tio.Subject(
            mr=tio.ScalarImage(mr_path),
            ct=tio.ScalarImage(ct_path)
        )
        
        # --- PREPROCESSING FIX: Align the MR to the CT's physical space ---
        # Create a transform that will resample any image onto the CT's grid
        resampler = tio.Resample(target=subject.ct)
        # Apply it to the MR image
        mr_aligned = resampler(subject.mr)

        # Create a new subject with the now-aligned MR and original CT
        aligned_subject = tio.Subject(
            mr=mr_aligned,
            ct=subject.ct
        )

        # 4. Apply the on-the-fly augmentation pipeline (if any) to the aligned subject
        if self.transform:
            aligned_subject = self.transform(aligned_subject)

        mr_vol_aug = aligned_subject.mr.data
        ct_vol_aug = aligned_subject.ct.data

        # Downsample the full 3D augmented volumes first
        mr_vol_downsampled = F.interpolate(mr_vol_aug.unsqueeze(0), size=self.target_size, mode='trilinear', align_corners=False).squeeze()
        ct_vol_downsampled = F.interpolate(ct_vol_aug.unsqueeze(0), size=self.target_size, mode='trilinear', align_corners=False).squeeze()

        # For simplicity, just takes the middle slice for contrastive learning
        mid_slice_idx = mr_vol_downsampled.shape[0] // 2
        
        mr_slice = mr_vol_downsampled[mid_slice_idx, :, :].unsqueeze(0)
        ct_slice = ct_vol_downsampled[mid_slice_idx, :, :].unsqueeze(0)
        
        return mr_slice, ct_slice

class VlearnDataset(Dataset):
    """
    A PyTorch Dataset that loads, aligns, and preprocesses MR/CT pairs.
    """
    def __init__(self, path_pairs, transform=None, target_size=(128, 128, 128)):
        self.path_pairs = path_pairs
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.path_pairs)

    def __getitem__(self, idx):
        # 1. Get the file paths for the requested index
        mr_path, ct_path = self.path_pairs[idx]

        try:
            # 2. Load the original images into a TorchIO Subject
            subject = tio.Subject(
                mr=tio.ScalarImage(mr_path),
                ct=tio.ScalarImage(ct_path)
            )

            # --- PREPROCESSING FIX: Align the MR to the CT's physical space ---
            # Create a transform that will resample any image onto the CT's grid
            resampler = tio.Resample(target=subject.ct)
            # Apply it to the MR image
            mr_aligned = resampler(subject.mr)

            # Create a new subject with the now-aligned MR and original CT
            aligned_subject = tio.Subject(
                mr=mr_aligned,
                ct=subject.ct
            )
            # -----------------------------------------------------------------

            # 4. Apply the on-the-fly augmentation pipeline (if any) to the aligned subject
            if self.transform:
                aligned_subject = self.transform(aligned_subject)

            # 5. Get the final tensors
            mr_vol = aligned_subject.mr.data
            ct_vol = aligned_subject.ct.data
            
            # 6. Downsample the volumes to the final target size for the model
            mr_downsampled = F.interpolate(mr_vol.unsqueeze(0), size=self.target_size, mode='trilinear', align_corners=False).squeeze(0)
            ct_downsampled = F.interpolate(ct_vol.unsqueeze(0), size=self.target_size, mode='trilinear', align_corners=False).squeeze(0)

            # 7. Return the aligned and processed tensors
            # Note: After resampling, spacing is implicitly the same, but you can return it if needed.
            # The new MR spacing will be identical to the CT spacing.
            final_spacing = aligned_subject.ct.spacing
            
            return mr_downsampled, ct_downsampled, final_spacing

        except Exception as e:
            print(f"Error loading or processing sample at index {idx}: {mr_path}, {ct_path}")
            print(f"Error: {e}")
            # Return a dummy sample or skip
            return self.__getitem__((idx + 1) % len(self))

