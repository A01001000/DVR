import torchio as tio
from pathlib import Path
import argparse
import os
import time
import torch
import numpy as np
import SimpleITK as sitk
import cv2
import glob
import re

def natural_sort_key(text):
    """Convert a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    
    return [tryint(c) for c in re.split(r'(\d+)', text)]

def natsorted(seq):
    """Sort sequence using natural ordering (numbers sorted numerically)"""
    return sorted(seq, key=natural_sort_key)

def create_histogram_landmarks(train_mr_paths, output_path):
    """
    Creates histogram landmarks from the training set MRIs.
    This version returns the landmarks dictionary and also saves it to a file.
    """
    print("Creating histogram landmarks from training data...")
    train_mr_paths_str = [str(p) for p in train_mr_paths]
    
    # Get the landmarks tensor from the train function
    landmarks_tensor = tio.HistogramStandardization.train(train_mr_paths_str)
    
    # Create the dictionary that the HistogramStandardization class expects
    landmarks_dict = {'MR': landmarks_tensor}  # Use 'MR' as key for MR modality
    
    # Manually save the correctly formatted dictionary to the specified path
    print(f"Attempting to manually save landmarks dictionary to {output_path}...")
    torch.save(landmarks_dict, output_path)
    print(f"Landmarks dictionary saved to {output_path}")
    
    # Return the dictionary object
    return landmarks_dict

def find_common_liver_z_range(ct_label, mr_label):
    """
    Find the overlapping Z-range (slice range) where both CT and MR have liver labels.
    Only crops in Z-dimension to remove non-overlapping superior/inferior slices.
    """
    if ct_label is None or mr_label is None:
        return None
    
    # Get binary masks for liver (assuming liver label is 1)
    ct_data = ct_label.data.squeeze().numpy()
    mr_data = mr_label.data.squeeze().numpy()
    
    # Find Z-range (slice range) for liver in both modalities
    ct_liver_slices = np.where(np.any(ct_data > 0, axis=(0, 1)))[0]
    mr_liver_slices = np.where(np.any(mr_data > 0, axis=(0, 1)))[0]
    
    if len(ct_liver_slices) == 0 or len(mr_liver_slices) == 0:
        return None
    
    # Get Z-range bounds
    ct_z_min, ct_z_max = np.min(ct_liver_slices), np.max(ct_liver_slices)
    mr_z_min, mr_z_max = np.min(mr_liver_slices), np.max(mr_liver_slices)
    
    # Find intersection in Z-dimension only
    common_z_min = max(ct_z_min, mr_z_min)
    common_z_max = min(ct_z_max, mr_z_max)
    
    # Add small padding in Z only
    z_padding = 2
    final_z_min = max(0, common_z_min - z_padding)
    final_z_max = min(ct_data.shape[2] - 1, common_z_max + z_padding)
    
    print(f"    Z-ranges: CT [{ct_z_min}:{ct_z_max}], MR [{mr_z_min}:{mr_z_max}] -> Common [{final_z_min}:{final_z_max}]")
    
    return (final_z_min, final_z_max)

def crop_to_common_z_range(subject, z_range):
    """
    Crop subject to common Z-range (slice range) only.
    Preserves full X and Y dimensions, only removes slices outside liver overlap.
    """
    if z_range is None:
        return subject
    
    z_min, z_max = z_range
    
    # Determine which image attribute to use
    if hasattr(subject, 'image'):
        image = subject.image
    elif hasattr(subject, 'MR'):
        image = subject.MR
    else:
        print(f"    Warning: Subject has no 'image' or 'MR' attribute. Available: {list(subject.keys())}")
        return subject
    
    # Calculate crop parameters: (left, right, anterior, posterior, inferior, superior)
    # We only crop in Z (superior/inferior), leave X/Y unchanged
    z_shape = image.shape[3]  # Z is the 4th dimension in TorchIO (Channel, X, Y, Z)
    
    crop_inferior = max(0, z_min)  # Remove slices below z_min, ensure non-negative
    crop_superior = max(0, z_shape - z_max - 1)  # Remove slices above z_max, ensure non-negative
    
    # Sanity check: ensure we don't crop more than available
    if crop_inferior + crop_superior >= z_shape:
        print(f"    Warning: Crop bounds would remove all slices! z_shape={z_shape}, crop_inf={crop_inferior}, crop_sup={crop_superior}")
        return subject
    
    crop_transform = tio.Crop(
        (0, 0,  # No X cropping (left, right)
         0, 0,  # No Y cropping (anterior, posterior)  
         crop_inferior, crop_superior)  # Only Z cropping (inferior, superior)
    )
    
    print(f"    Cropping Z: removing {crop_inferior} inferior + {crop_superior} superior slices")
    
    return crop_transform(subject)

def mask_liver_to_intersection(ct_subject, mr_subject):
    """
    Create intersection masks for liver labels, handling different anatomical coverage.
    
    IMPORTANT: Both subjects must be in the same space (spacing/shape) before calling this.
    """
    if ct_subject.label is None or mr_subject.label is None:
        return ct_subject, mr_subject
    
    # Get label data - ensure they're in the same space
    ct_label_data = ct_subject.label.data.squeeze()
    mr_label_data = mr_subject.label.data.squeeze()
    
    # Verify they have the same shape (they should after spatial transforms)
    if ct_label_data.shape != mr_label_data.shape:
        print(f"    Warning: Different label shapes - CT: {ct_label_data.shape}, MR: {mr_label_data.shape}")
        print("    Attempting spatial alignment...")
        
        try:
            # Try to align MR label to CT space
            mr_label_subject = tio.Subject(label=mr_subject.label)
            alignment_transform = tio.Compose([
                tio.Resample(ct_subject.image.spacing),
                tio.CropOrPad(ct_label_data.shape)
            ])
            mr_label_aligned = alignment_transform(mr_label_subject)
            mr_label_data = mr_label_aligned.label.data.squeeze()
            
            print(f"    ✓ Labels aligned: CT {ct_label_data.shape}, MR {mr_label_data.shape}")
            
            # Update MR subject with aligned label
            mr_subject = tio.Subject(
                MR=mr_subject.MR,
                label=mr_label_aligned.label
            )
            
        except Exception as e:
            print(f"    Failed to align labels: {e}")
            print("    Skipping intersection - keeping original labels")
            return ct_subject, mr_subject
    
    # Create intersection mask (where both have liver)
    intersection_mask = (ct_label_data > 0) & (mr_label_data > 0)
    intersection_voxels = torch.sum(intersection_mask).item()
    
    if intersection_voxels == 0:
        print("    Warning: No liver intersection found - keeping original labels")
        return ct_subject, mr_subject
    
    # Calculate coverage statistics
    ct_liver_voxels = torch.sum(ct_label_data > 0).item()
    mr_liver_voxels = torch.sum(mr_label_data > 0).item()
    
    ct_intersection_ratio = intersection_voxels / ct_liver_voxels if ct_liver_voxels > 0 else 0
    mr_intersection_ratio = intersection_voxels / mr_liver_voxels if mr_liver_voxels > 0 else 0
    
    print(f"    Liver intersection: {intersection_voxels} voxels")
    print(f"    CT coverage: {ct_intersection_ratio:.1%}, MR coverage: {mr_intersection_ratio:.1%}")
    
    # Only apply intersection if it's reasonable (at least 20% overlap)
    if ct_intersection_ratio < 0.2 or mr_intersection_ratio < 0.2:
        print("    Warning: Low liver overlap - keeping original labels for better coverage")
        return ct_subject, mr_subject
    
    # Apply intersection to both labels
    ct_new_label = ct_label_data.clone()
    mr_new_label = mr_label_data.clone()
    
    ct_new_label[~intersection_mask] = 0
    mr_new_label[~intersection_mask] = 0
    
    # Create new subjects with intersection labels, preserving original spacing/affine
    ct_subject_new = tio.Subject(
        image=ct_subject.image,
        label=tio.LabelMap(tensor=ct_new_label.unsqueeze(0), affine=ct_subject.label.affine)
    )
    
    # For MR subject, preserve the MR key and spacing
    mr_subject_new = tio.Subject(
        MR=mr_subject.MR,
        label=tio.LabelMap(tensor=mr_new_label.unsqueeze(0), affine=mr_subject.label.affine)
    )
    
    return ct_subject_new, mr_subject_new

def analyze_liver_coverage(ct_subject, mr_subject, patient_id):
    """
    Analyze and report liver coverage differences between CT and MR.
    """
    if ct_subject.label is None or mr_subject.label is None:
        print(f"    - {patient_id}: Missing labels, skipping analysis")
        return
    
    ct_liver_voxels = torch.sum(ct_subject.label.data > 0).item()
    mr_liver_voxels = torch.sum(mr_subject.label.data > 0).item()
    
    # Check if labels have compatible shapes for intersection calculation
    ct_data = ct_subject.label.data.squeeze()
    mr_data = mr_subject.label.data.squeeze()
    
    if ct_data.shape != mr_data.shape:
        print(f"    - {patient_id}: CT liver: {ct_liver_voxels} voxels, MR liver: {mr_liver_voxels} voxels")
        print(f"    - {patient_id}: Shape mismatch - CT: {ct_data.shape}, MR: {mr_data.shape} - cannot calculate intersection")
        return
    
    # Calculate intersection only if shapes match
    intersection_voxels = torch.sum((ct_data > 0) & (mr_data > 0)).item()
    
    ct_only_voxels = ct_liver_voxels - intersection_voxels
    mr_only_voxels = mr_liver_voxels - intersection_voxels
    
    overlap_ratio = intersection_voxels / max(ct_liver_voxels, mr_liver_voxels) if max(ct_liver_voxels, mr_liver_voxels) > 0 else 0
    
    print(f"    - {patient_id}: CT liver: {ct_liver_voxels} voxels, MR liver: {mr_liver_voxels} voxels")
    print(f"    - {patient_id}: Overlap: {intersection_voxels} voxels ({overlap_ratio:.2%})")
    print(f"    - {patient_id}: CT-only: {ct_only_voxels}, MR-only: {mr_only_voxels}")
    
    return {
        'ct_voxels': ct_liver_voxels,
        'mr_voxels': mr_liver_voxels,
        'intersection': intersection_voxels,
        'overlap_ratio': overlap_ratio
    }

def verify_output_properties(ct_subject, mr_subject, patient_id, target_spacing=(2, 2, 2), target_size=(192, 192, 192)):
    """
    Verify that the processed subjects have the correct spacing and size.
    """
    print(f"    - Verification for {patient_id}:")
    
    # Check CT properties
    ct_spacing = tuple(ct_subject.image.spacing)
    ct_shape = tuple(ct_subject.image.shape[1:])  # Remove channel dimension
    print(f"      CT: spacing={ct_spacing}, shape={ct_shape}")
    
    # Check MR properties
    mr_spacing = tuple(mr_subject.MR.spacing)
    mr_shape = tuple(mr_subject.MR.shape[1:])  # Remove channel dimension
    print(f"      MR: spacing={mr_spacing}, shape={mr_shape}")
    
    # Verify against targets
    spacing_ok = (abs(ct_spacing[0] - target_spacing[0]) < 0.01 and 
                  abs(ct_spacing[1] - target_spacing[1]) < 0.01 and
                  abs(ct_spacing[2] - target_spacing[2]) < 0.01 and
                  abs(mr_spacing[0] - target_spacing[0]) < 0.01 and
                  abs(mr_spacing[1] - target_spacing[1]) < 0.01 and
                  abs(mr_spacing[2] - target_spacing[2]) < 0.01)
    
    size_ok = (ct_shape == target_size and mr_shape == target_size)
    
    if spacing_ok and size_ok:
        print(f"      ✓ All properties correct!")
    else:
        if not spacing_ok:
            print(f"      ✗ Spacing issue detected!")
        if not size_ok:
            print(f"      ✗ Size issue detected!")
    
    return spacing_ok and size_ok

def convert_chaos_patient_data(patient_dir, dicom_root, png_root):
    """
    Converts original DICOM and PNG data for a single CHAOS patient to NIfTI format
    using overlapping slice matching AND proper spatial alignment for labels.
    """
    print(f"    - Converting original DICOM/PNG data (slice matching + spatial alignment)...")

    # --- Step 1: Find all file paths using the provided root directories ---
    dicom_root_path = Path(dicom_root)
    png_root_path = Path(png_root)
    
    # Extract split and patient info from patient_dir
    split_name = patient_dir.parent.name  # Train or Test
    patient_id = patient_dir.name
    
    ct_dicom_path = dicom_root_path / split_name / patient_id / 'CT'
    mr_dicom_path = dicom_root_path / split_name / patient_id / 'MR'
    
    # PNG paths (no Train/Test level - directly under patient_id)
    # IMPORTANT: PNG segmentations only exist for Test patients
    ct_png_path = png_root_path / patient_id / 'CT'
    mr_png_path = png_root_path / patient_id / 'MR'
    
    # Check if PNG segmentations exist (only for Test patients)
    has_png_segmentations = ct_png_path.exists() and mr_png_path.exists()
    
    # Debug: Check what paths exist
    if split_name == 'Test':
        paths_info = [
            ("CT DICOM", ct_dicom_path, ct_dicom_path.exists()),
            ("CT PNG", ct_png_path, ct_png_path.exists()),
            ("MR DICOM", mr_dicom_path, mr_dicom_path.exists()),
            ("MR PNG", mr_png_path, mr_png_path.exists())
        ]
        
        missing_paths = []
        for name, path, exists in paths_info:
            if exists:
                print(f"      - ✓ Found {name}: {path}")
            else:
                print(f"      - ✗ Missing {name}: {path}")
                missing_paths.append(name)
        
        # For Test patients, we need both DICOM and PNG
        if missing_paths:
            print(f"      - Skipping conversion: missing paths: {', '.join(missing_paths)}")
            return False
    else:
        # For Train patients, only check DICOM (no PNG segmentations available)
        paths_info = [
            ("CT DICOM", ct_dicom_path, ct_dicom_path.exists()),
            ("MR DICOM", mr_dicom_path, mr_dicom_path.exists())
        ]
        
        missing_paths = []
        for name, path, exists in paths_info:
            if exists:
                print(f"      - ✓ Found {name}: {path}")
            else:
                print(f"      - ✗ Missing {name}: {path}")
                missing_paths.append(name)
        
        print(f"      - Train patient: No PNG segmentations expected (CHAOS dataset structure)")
        
        # For Train patients, we only need DICOM
        if missing_paths:
            print(f"      - Skipping conversion: missing paths: {', '.join(missing_paths)}")
            return False

    # --- Step 2: Find overlapping slices (as per original CHAOS approach) ---
    ct_dicom_files = natsorted(glob.glob(os.path.join(ct_dicom_path, '*.dcm')))
    mr_dicom_files = natsorted(glob.glob(os.path.join(mr_dicom_path, '*.dcm')))
    
    if has_png_segmentations:
        # For Test patients: find overlapping slices across all modalities
        ct_png_files = natsorted(glob.glob(os.path.join(ct_png_path, '*.png')))
        mr_png_files = natsorted(glob.glob(os.path.join(mr_png_path, '*.png')))
        
        print(f"      - File counts: CT DICOM={len(ct_dicom_files)}, CT PNG={len(ct_png_files)}, MR DICOM={len(mr_dicom_files)}, MR PNG={len(mr_png_files)}")
        
        # Find common slice count across all modalities (as per CHAOS rules)
        common_slice_count = min(len(ct_dicom_files), len(ct_png_files), len(mr_dicom_files), len(mr_png_files))
        
        if common_slice_count == 0:
            print("      - Skipping conversion: no common slices found across all modalities.")
            return False
            
        print(f"      - Found {common_slice_count} common slices across CT and MR based on file count.")
        
        # Trim all file lists to the common slice count (keep only overlapping slices)
        ct_dicom_files = ct_dicom_files[:common_slice_count]
        ct_png_files = ct_png_files[:common_slice_count]
        mr_dicom_files = mr_dicom_files[:common_slice_count]
        mr_png_files = mr_png_files[:common_slice_count]
    else:
        # For Train patients: no PNG files, just use DICOM files
        print(f"      - File counts: CT DICOM={len(ct_dicom_files)}, MR DICOM={len(mr_dicom_files)} (no PNG segmentations)")
        
        # For Train patients, we don't have overlapping slice constraints
        # Just process all available DICOM slices
        common_slice_count = min(len(ct_dicom_files), len(mr_dicom_files))
        
        if common_slice_count == 0:
            print("      - Skipping conversion: no DICOM files found.")
            return False
            
        print(f"      - Found {common_slice_count} common slices between CT and MR DICOM.")
        
        # Trim DICOM files to common count
        ct_dicom_files = ct_dicom_files[:common_slice_count]
        mr_dicom_files = mr_dicom_files[:common_slice_count]

    # --- Step 3: Create output directories ---
    ct_nifti_img_out_dir = patient_dir / 'CT'
    mr_nifti_img_out_dir = patient_dir / 'MR'
    
    # Only create segmentation directories for Test patients (who have PNG segmentations)
    if has_png_segmentations:
        ct_nifti_seg_out_dir = patient_dir / 'CT_seg'
        mr_nifti_seg_out_dir = patient_dir / 'MR_seg'
        
        for out_dir in [ct_nifti_img_out_dir, ct_nifti_seg_out_dir, mr_nifti_img_out_dir, mr_nifti_seg_out_dir]:
            out_dir.mkdir(exist_ok=True, parents=True)
            
        ct_seg_nifti_path = ct_nifti_seg_out_dir / f'{patient_id}_CT_seg.nii.gz'
        mr_seg_nifti_path = mr_nifti_seg_out_dir / f'{patient_id}_MR_seg.nii.gz'
    else:
        # Train patients: only image directories
        for out_dir in [ct_nifti_img_out_dir, mr_nifti_img_out_dir]:
            out_dir.mkdir(exist_ok=True, parents=True)
    
    ct_img_nifti_path = ct_nifti_img_out_dir / f'{patient_id}_CT.nii.gz'
    mr_img_nifti_path = mr_nifti_img_out_dir / f'{patient_id}_MR.nii.gz'

    try:
        # --- Step 4: Convert CT DICOM to NIfTI (using only overlapping slices) ---
        print(f"      - Converting CT DICOM series (using {len(ct_dicom_files)} slices)...")
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(ct_dicom_files)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        ct_image = reader.Execute()
        sitk.WriteImage(ct_image, str(ct_img_nifti_path))
        
        # --- Step 5: Convert CT PNG labels with spatial alignment (only for Test patients) ---
        if has_png_segmentations:
            print(f"      - Converting CT labels with spatial alignment (using {len(ct_png_files)} slices)...")
            
            # Read overlapping PNG slices only
            ct_slice_arrays = []
            for file in ct_png_files:
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Failed to read {file}")
                ct_slice_arrays.append(img)
            
            # Stack slices into a 3D volume (Z, Y, X)
            ct_volume_array = np.stack(ct_slice_arrays, axis=0)
            ct_mask_array = (ct_volume_array > 0).astype(np.uint8)  # CT thresholding
            
            # Create SimpleITK image from the numpy array
            ct_label_image = sitk.GetImageFromArray(ct_mask_array)
            
            # **CRITICAL**: Copy spatial information from reference CT image
            ct_label_image.CopyInformation(ct_image)
            
            # Resample to ensure perfect alignment to reference grid
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(ct_image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            ct_aligned_label = resampler.Execute(ct_label_image)
            
            sitk.WriteImage(ct_aligned_label, str(ct_seg_nifti_path))
        
        # --- Step 6: Convert MR DICOM to NIfTI (using only overlapping slices) ---
        print(f"      - Converting MR DICOM series (using {len(mr_dicom_files)} slices)...")
        reader.SetFileNames(mr_dicom_files)
        mr_image = reader.Execute()
        sitk.WriteImage(mr_image, str(mr_img_nifti_path))
        
        # --- Step 7: Convert MR PNG labels with spatial alignment (only for Test patients) ---
        if has_png_segmentations:
            print(f"      - Converting MR labels with spatial alignment (using {len(mr_png_files)} slices)...")
            
            # Read overlapping PNG slices only
            mr_slice_arrays = []
            for file in mr_png_files:
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Failed to read {file}")
                mr_slice_arrays.append(img)
            
            # Stack slices into a 3D volume (Z, Y, X)
            mr_volume_array = np.stack(mr_slice_arrays, axis=0)
            
            # Apply MR-specific liver thresholding (values 55-70)
            liver_mask = (mr_volume_array >= 55) & (mr_volume_array <= 70)
            mr_mask_array = np.zeros_like(mr_volume_array, dtype=np.uint8)
            mr_mask_array[liver_mask] = 1
            
            # Create SimpleITK image from the numpy array
            mr_label_image = sitk.GetImageFromArray(mr_mask_array)
            
            # **CRITICAL**: Copy spatial information from reference MR image
            mr_label_image.CopyInformation(mr_image)
            
            # Resample to ensure perfect alignment to reference grid
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(mr_image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            mr_aligned_label = resampler.Execute(mr_label_image)
            
            sitk.WriteImage(mr_aligned_label, str(mr_seg_nifti_path))
            
            print(f"      - ✅ Successfully converted patient {patient_id} with {common_slice_count} slices and segmentations")
        else:
            print(f"      - ✅ Successfully converted patient {patient_id} with {common_slice_count} slices (images only - no segmentations for Train patients)")
        
        return True
        
    except Exception as e:
        print(f"      - ✗ Failed to convert patient {patient_id}: {e}")
        return False

def preprocess_dataset(dicom_root, png_root, output_dir, final_size=(192, 192, 192), target_spacing=(2, 2, 2), 
                      liver_handling='analyze', max_patients=None):
    """
    Main function to preprocess the entire dataset using only TorchIO.
    This pipeline discovers patients from the dicom_root, converts them to NIfTI in the output_dir,
    and then applies all further preprocessing steps.
    """
    dicom_root_path = Path(dicom_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Convert all data from DICOM/PNG to NIfTI first ---
    # This creates the base NIfTI dataset in the output directory that we will then process.
    print("--- Running initial data conversion to NIfTI format ---")
    conversion_successful = False
    
    for split in ['Train', 'Test']:
        split_dicom_path = dicom_root_path / split
        if not split_dicom_path.is_dir():
            print(f"Source split '{split}' not found in DICOM root, skipping.")
            continue
        
        # Discover patients from the DICOM directory structure
        source_patient_dirs = sorted([p for p in split_dicom_path.iterdir() if p.is_dir()])
        
        for source_patient_dir in source_patient_dirs:
            patient_id = source_patient_dir.name
            print(f"  -> Converting patient: {patient_id}")
            
            # Define the output directory for this patient's NIfTI files
            patient_output_dir = output_path / split / patient_id
            patient_output_dir.mkdir(parents=True, exist_ok=True)
            
            # The conversion function now takes the output dir as the patient_dir
            success = convert_chaos_patient_data(patient_output_dir, dicom_root, png_root)
            if success:
                conversion_successful = True
                print(f"      - ✓ Successfully converted patient {patient_id}")
            else:
                print(f"      - ✗ Failed to convert patient {patient_id}")

    if not conversion_successful:
        print("ERROR: No patients were successfully converted from DICOM/PNG format.")
        print("Please check:")
        print(f"  - DICOM root path: {dicom_root}")
        print(f"  - PNG root path: {png_root}")
        print("  - Expected DICOM structure: DICOM_root/Train/[PatientID]/CT/*.dcm")
        print("  - Expected DICOM structure: DICOM_root/Train/[PatientID]/MR/*.dcm")
        print("  - Expected PNG structure: PNG_root/[PatientID]/CT/*.png")
        print("  - Expected PNG structure: PNG_root/[PatientID]/MR/*.png")
        return

    # --- Step 2: Create Histogram Landmarks from the newly created Training Set NIfTIs ---
    train_mr_files = list((output_path / 'Train').glob('*/MR/*.nii.gz'))
    landmarks_path = output_path / 'mr_landmarks.pt'
    
    print(f"\nFound {len(train_mr_files)} MR files in the processed Train set for landmark creation.")
    
    if landmarks_path.is_file():
        print(f"Loading existing landmarks from {landmarks_path}")
        landmarks_dict = torch.load(landmarks_path)
        if not isinstance(landmarks_dict, dict):
            landmarks_dict = {'MR': landmarks_dict}
    elif train_mr_files:
        print(f"Landmarks file not found. Creating it...")
        landmarks_dict = create_histogram_landmarks(train_mr_files, landmarks_path)
    else:
        print("Warning: No MR files found in processed Train set. Using Test set for landmarks...")
        test_mr_files = list((output_path / 'Test').glob('*/MR/*.nii.gz'))
        if test_mr_files:
            landmarks_dict = create_histogram_landmarks(test_mr_files, landmarks_path)
        else:
            raise FileNotFoundError("Cannot create landmarks: No MR files found in processed Train or Test sets.")

    # --- Step 3: Define the Preprocessing Transforms ---
    
    # For CT: Clamp HU values, then rescale to 0-1
    ct_normalization = tio.Compose([
        tio.Clamp(out_min=-200, out_max=300),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])
    
    # For MR: Use histogram standardization based on the training set
    mr_normalization = tio.HistogramStandardization(landmarks=landmarks_dict)

    # Define spatial transforms - CRITICAL: Use ResampleOrCrop to ensure exact size AND spacing
    ct_spatial_transforms = tio.Compose([
        tio.Resample(target_spacing),
        tio.CropOrPad(final_size),
    ])
    
    mr_spatial_transforms = tio.Compose([
        tio.Resample(target_spacing),
        tio.CropOrPad(final_size),
    ])
    
    # --- Step 4: Process Each Split (Train, Test) ---
    for split in ['Train', 'Test']:
        split_path = output_path / split
        if not split_path.is_dir():
            print(f"Processed split '{split}' not found, skipping.")
            continue
            
        print(f"\n--- Processing {split} set ---")
        
        # Patient directories are now in the output path
        patient_dirs = sorted([p for p in split_path.iterdir() if p.is_dir()])
        
        if max_patients is not None:
            patient_dirs = patient_dirs[:max_patients]
            print(f"Processing only first {len(patient_dirs)} patients (max_patients={max_patients})")
        
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            print(f"  -> Patient: {patient_id}")
            
            try:
                # Paths now point to the NIfTI files in the output directory
                ct_img_path_list = list(patient_dir.glob('CT/*.nii.gz'))
                mr_img_path_list = list(patient_dir.glob('MR/*.nii.gz'))
                
                if not ct_img_path_list or not mr_img_path_list:
                    print(f"    - Skipping patient {patient_id}: NIfTI files not found after conversion attempt.")
                    continue
                
                ct_img_path = ct_img_path_list[0]
                mr_img_path = mr_img_path_list[0]
                
                ct_label_paths = list(patient_dir.glob('CT_seg/*.nii.gz'))
                mr_label_paths = list(patient_dir.glob('MR_seg/*.nii.gz'))
                ct_label_path = ct_label_paths[0] if ct_label_paths else None
                mr_label_path = mr_label_paths[0] if mr_label_paths else None

                # --- Load data into TorchIO Subjects ---
                fixed_subject = tio.Subject(
                    image=tio.ScalarImage(ct_img_path),
                    label=tio.LabelMap(ct_label_path) if ct_label_path else None
                )
                
                moving_subject = tio.Subject(
                    MR=tio.ScalarImage(mr_img_path),
                    label=tio.LabelMap(mr_label_path) if mr_label_path else None
                )
                
                # --- CRITICAL: Apply ToCanonical first ---
                print("    - Applying canonical orientation...")
                canonical_transform = tio.ToCanonical()
                fixed_subject = canonical_transform(fixed_subject)
                moving_subject = canonical_transform(moving_subject)
                
                # Debug: Check image properties after canonical
                print(f"    - Debug: CT shape after canonical: {fixed_subject.image.shape}, spacing: {fixed_subject.image.spacing}")
                print(f"    - Debug: MR shape after canonical: {moving_subject.MR.shape}, spacing: {moving_subject.MR.spacing}")
                
                # Check for potential orientation/slice ordering issues
                ct_data = fixed_subject.image.data.squeeze()
                mr_data = moving_subject.MR.data.squeeze()
                print(f"    - Debug: CT intensity range: {ct_data.min():.1f} to {ct_data.max():.1f}")
                print(f"    - Debug: MR intensity range: {mr_data.min():.1f} to {mr_data.max():.1f}")
                
                # Check if MR data looks corrupted (all same value)
                if torch.all(mr_data == mr_data.flatten()[0]):
                    print(f"    - Warning: MR image appears to have constant intensity!")
                
                # Debug label shapes if they exist
                if fixed_subject.label and moving_subject.label:
                    ct_label_data = fixed_subject.label.data.squeeze()
                    mr_label_data = moving_subject.label.data.squeeze()
                    print(f"    - Debug: CT label shape: {ct_label_data.shape}, non-zero: {torch.sum(ct_label_data > 0).item()}")
                    print(f"    - Debug: MR label shape: {mr_label_data.shape}, non-zero: {torch.sum(mr_label_data > 0).item()}")
                    
                    # Check for potential slice ordering reversal
                    if ct_label_data.shape == mr_label_data.shape:
                        # Compare liver center of mass to detect reversal
                        ct_liver_slices = torch.where(torch.any(ct_label_data > 0, dim=(0, 1)))[0]
                        mr_liver_slices = torch.where(torch.any(mr_label_data > 0, dim=(0, 1)))[0]
                        
                        if len(ct_liver_slices) > 0 and len(mr_liver_slices) > 0:
                            ct_liver_center = torch.mean(ct_liver_slices.float()).item()
                            mr_liver_center = torch.mean(mr_liver_slices.float()).item()
                            ct_liver_span = ct_liver_slices[-1] - ct_liver_slices[0]
                            mr_liver_span = mr_liver_slices[-1] - mr_liver_slices[0]
                            
                            print(f"    - Debug: Liver centers - CT: {ct_liver_center:.1f}, MR: {mr_liver_center:.1f}")
                            print(f"    - Debug: Liver spans - CT: {ct_liver_span}, MR: {mr_liver_span}")
                            
                            # Check if orientations are drastically different (possible reversal)
                            center_diff = abs(ct_liver_center - mr_liver_center)
                            max_center = max(ct_liver_center, mr_liver_center)
                            if center_diff > 0.7 * max_center:  # Centers are far apart
                                print(f"    - Warning: Possible slice ordering reversal detected!")
                                print(f"    - Consider flipping MR slice order for better alignment")
                
                # --- Skip duplicate normalization - will be done in final pipeline ---
                print("    - Skipping early normalization (will be applied in final pipeline)")
                
                if liver_handling == 'crop' and fixed_subject.label and moving_subject.label:
                    print("    - Finding common liver Z-range...")
                    # For Z-range analysis, we need compatible shapes - do a quick resample just for analysis
                    temp_mr_resampled = tio.Resample(fixed_subject.image.spacing)(moving_subject)
                    common_z_range = find_common_liver_z_range(fixed_subject.label, temp_mr_resampled.label)
                    if common_z_range:
                        # CRITICAL: Crop BEFORE any spatial transforms to avoid spacing issues
                        print("    - Cropping to common Z-range (before spatial alignment)...")
                        fixed_subject = crop_to_common_z_range(fixed_subject, common_z_range)
                        moving_subject = crop_to_common_z_range(moving_subject, common_z_range)
                        print(f"    - Z-range cropping completed")
                
                # --- Apply spatial alignment AFTER cropping with enhanced verification ---
                print("    - Applying spatial alignment...")
                # First, resample MR to match CT spacing to ensure compatible tensor operations
                if fixed_subject.label and moving_subject.label:
                    try:
                        # After Z-cropping, we need to ensure both images have compatible shapes for alignment
                        ct_shape = fixed_subject.image.shape[1:]  # Skip channel dimension
                        mr_shape = moving_subject.MR.shape[1:]
                        
                        print(f"      Post-crop shapes: CT {ct_shape}, MR {mr_shape}")
                        
                        # Resample MR (moving) to CT (fixed) spacing for compatible shapes
                        alignment_transform = tio.Resample(fixed_subject.image.spacing)
                        moving_subject_aligned = alignment_transform(moving_subject)
                        
                        # Verify alignment worked correctly
                        new_mr_shape = moving_subject_aligned.MR.shape[1:]
                        print(f"      Post-alignment MR shape: {new_mr_shape}")
                        
                        # Update moving_subject with aligned version for liver analysis
                        moving_subject = moving_subject_aligned
                        print(f"    - Spatial alignment successful")
                    except Exception as e:
                        print(f"    - Warning: Spatial alignment failed ({e}). Proceeding with original spacing.")
                        # Continue without spatial alignment if it fails
                
                # --- Analyze liver coverage AFTER spatial alignment ---
                if liver_handling == 'analyze':
                    liver_stats = analyze_liver_coverage(fixed_subject, moving_subject, patient_id)
                
                # --- Handle mismatched liver labels AFTER spatial alignment ---
                if liver_handling == 'intersection' and fixed_subject.label and moving_subject.label:
                    print("    - Creating intersection liver masks...")
                    try:
                        # Store original spacings before intersection processing
                        ct_original_spacing = fixed_subject.image.spacing
                        mr_original_spacing = moving_subject.MR.spacing
                        
                        # Labels should now be spatially aligned from the previous step
                        # But let's double-check and align labels specifically if needed
                        ct_label_shape = fixed_subject.label.data.squeeze().shape
                        mr_label_shape = moving_subject.label.data.squeeze().shape
                        
                        if ct_label_shape != mr_label_shape:
                            print(f"    - Aligning label spaces: CT {ct_label_shape} -> MR {mr_label_shape}")
                            # Apply the same spatial alignment to labels that was applied to images
                            label_alignment_transform = tio.Resample(fixed_subject.image.spacing)
                            
                            # Create temporary subject with just the label for resampling
                            temp_label_subject = tio.Subject(label=moving_subject.label)
                            temp_label_aligned = label_alignment_transform(temp_label_subject)
                            
                            # Update moving_subject with aligned label
                            moving_subject = tio.Subject(
                                MR=moving_subject.MR,
                                label=temp_label_aligned.label
                            )
                            print(f"    - Label alignment completed")
                        
                        # SKIP INTERSECTION FOR NOW - it's causing subject corruption
                        print("    - Skipping intersection processing to avoid subject corruption")
                        # Now perform intersection with aligned labels
                        # fixed_subject, moving_subject = mask_liver_to_intersection(fixed_subject, moving_subject)
                        
                        # Report liver statistics instead
                        ct_liver_voxels = torch.sum(fixed_subject.label.data > 0).item()
                        mr_liver_voxels = torch.sum(moving_subject.label.data > 0).item()
                        ct_label_data = fixed_subject.label.data.squeeze()
                        mr_label_data = moving_subject.label.data.squeeze()
                        
                        if ct_label_data.shape == mr_label_data.shape:
                            intersection_mask = (ct_label_data > 0) & (mr_label_data > 0)
                            intersection_voxels = torch.sum(intersection_mask).item()
                            ct_coverage = intersection_voxels / ct_liver_voxels if ct_liver_voxels > 0 else 0
                            mr_coverage = intersection_voxels / mr_liver_voxels if mr_liver_voxels > 0 else 0
                            print(f"    Liver analysis: CT={ct_liver_voxels}, MR={mr_liver_voxels}, intersection={intersection_voxels}")
                            print(f"    Coverage: CT {ct_coverage:.1%}, MR {mr_coverage:.1%}")
                        
                        # CRITICAL: DO NOT re-align after intersection since we skipped it
                        # The images should already be aligned from earlier steps
                        
                    except Exception as e:
                        print(f"    Warning: Liver intersection failed ({e}). Proceeding without intersection.")
                        # Continue without intersection
                
                # --- CRITICAL: Final spatial verification before normalization ---
                print("    - Verifying spatial consistency before normalization...")
                try:
                    # Check if images are in the same space
                    ct_spacing = fixed_subject.image.spacing
                    mr_spacing = moving_subject.MR.spacing
                    ct_shape = fixed_subject.image.shape[1:]  # Skip channel dimension
                    mr_shape = moving_subject.MR.shape[1:]   # Skip channel dimension
                    
                    print(f"      CT: spacing={tuple(ct_spacing)}, shape={tuple(ct_shape)}")
                    print(f"      MR: spacing={tuple(mr_spacing)}, shape={tuple(mr_shape)}")
                    
                    # Convert to tensors for comparison
                    ct_spacing_tensor = torch.tensor(ct_spacing)
                    mr_spacing_tensor = torch.tensor(mr_spacing)
                    
                    # Check for mismatches
                    spacing_mismatch = not torch.allclose(ct_spacing_tensor, mr_spacing_tensor, atol=1e-3)
                    shape_mismatch = ct_shape != mr_shape
                    
                    if spacing_mismatch or shape_mismatch:
                        print(f"      ⚠️  Space mismatch detected - forcing final alignment...")
                        print(f"         Spacing mismatch: {spacing_mismatch}, Shape mismatch: {shape_mismatch}")
                        
                        # Force MR to match CT space exactly (both spacing AND shape)
                        final_alignment = tio.Compose([
                            tio.Resample(target=ct_spacing),
                            tio.CropOrPad(target_shape=ct_shape)
                        ])
                        
                        # Apply to MR subject
                        moving_subject_realigned = final_alignment(moving_subject)
                        moving_subject = moving_subject_realigned
                        
                        # Verify the fix worked
                        new_mr_spacing = moving_subject.MR.spacing
                        new_mr_shape = moving_subject.MR.shape[1:]
                        print(f"      ✓ Final alignment completed:")
                        print(f"         New MR: spacing={tuple(new_mr_spacing)}, shape={tuple(new_mr_shape)}")
                    else:
                        print(f"      ✓ Images already in same space")
                        
                except Exception as e:
                    print(f"      ⚠️  Spatial verification failed: {e}")
                    print(f"      Attempting emergency alignment...")
                    try:
                        # Emergency alignment - force MR to exactly match CT space
                        ct_spacing = fixed_subject.image.spacing
                        ct_shape = fixed_subject.image.shape[1:]  # Skip channel dimension
                        
                        emergency_align = tio.Compose([
                            tio.Resample(target=ct_spacing),
                            tio.CropOrPad(target_shape=ct_shape)
                        ])
                        
                        moving_subject = emergency_align(moving_subject)
                        
                        # Verify emergency fix
                        final_mr_spacing = moving_subject.MR.spacing
                        final_mr_shape = moving_subject.MR.shape[1:]
                        print(f"      ✓ Emergency alignment completed:")
                        print(f"         Final MR: spacing={tuple(final_mr_spacing)}, shape={tuple(final_mr_shape)}")
                        
                    except Exception as e2:
                        print(f"      ✗ Emergency alignment failed: {e2}")
                        # This should trigger the fallback processing
                
                # --- Apply the Full Pipeline ---
                print("    - Normalizing and resizing...")
                start_time = time.time()

                # 1. Apply intensity normalization (only once, here in the final pipeline)
                # Apply normalization to the entire subjects
                fixed_normalized_subject = tio.Subject(
                    image=ct_normalization(fixed_subject.image),
                    label=fixed_subject.label
                )
                moving_normalized_subject = mr_normalization(moving_subject)
                
                # 2. Apply spatial transforms (both CT and MR to 2x2x2mm and 192³)
                # CT: Resample to isotropic + resize to 192³
                fixed_processed_subject = ct_spatial_transforms(fixed_normalized_subject)
                # MR: Resample to 2x2x2mm + resize to 192³
                moving_processed_subject = mr_spatial_transforms(moving_normalized_subject)
                
                # Both are now at 2x2x2mm spacing and 192x192x192 size

                print(f"    - Processing done in {time.time() - start_time:.2f}s")
                
                # --- Verify output properties ---
                verify_output_properties(fixed_processed_subject, moving_processed_subject, patient_id)
                
                # --- Save the preprocessed data ---
                patient_output_dir = output_path / split / patient_id
                
                # Save images (now both are 192x192x192 with 2x2x2mm isotropic spacing)
                (patient_output_dir / 'CT').mkdir(parents=True, exist_ok=True)
                (patient_output_dir / 'MR').mkdir(parents=True, exist_ok=True)
                fixed_processed_subject.image.save(patient_output_dir / 'CT' / ct_img_path.name)
                moving_processed_subject.MR.save(patient_output_dir / 'MR' / mr_img_path.name)
                
                # Save labels (transformed to match images)
                if fixed_processed_subject.label:
                    (patient_output_dir / 'CT_seg').mkdir(parents=True, exist_ok=True)
                    fixed_processed_subject.label.save(patient_output_dir / 'CT_seg' / ct_label_path.name)
                
                if moving_processed_subject.label:
                    (patient_output_dir / 'MR_seg').mkdir(parents=True, exist_ok=True)
                    moving_processed_subject.label.save(patient_output_dir / 'MR_seg' / mr_label_path.name)

            except RuntimeError as e:
                if "tensor" in str(e).lower() or "shape" in str(e).lower():
                    print(f"    - Tensor/shape error for {patient_id}: {e}")
                    print(f"    - Attempting fallback processing without liver analysis...")
                    try:
                        # Fallback: process without liver-specific operations
                        fallback_fixed = canonical_transform(tio.Subject(
                            image=tio.ScalarImage(ct_img_path),
                            label=tio.LabelMap(ct_label_path) if ct_label_path else None
                        ))
                        fallback_moving = canonical_transform(tio.Subject(
                            MR=tio.ScalarImage(mr_img_path),
                            label=tio.LabelMap(mr_label_path) if mr_label_path else None
                        ))
                        
                        # Apply spatial transforms only
                        fixed_processed_subject = ct_spatial_transforms(fallback_fixed)
                        moving_processed_subject = mr_spatial_transforms(fallback_moving)
                        
                        # Save results
                        patient_output_dir = output_path / split / patient_id
                        (patient_output_dir / 'CT').mkdir(parents=True, exist_ok=True)
                        (patient_output_dir / 'MR').mkdir(parents=True, exist_ok=True)
                        
                        fixed_processed_subject.image.save(patient_output_dir / 'CT' / ct_img_path.name)
                        moving_processed_subject.MR.save(patient_output_dir / 'MR' / mr_img_path.name)
                        
                        if fixed_processed_subject.label:
                            (patient_output_dir / 'CT_seg').mkdir(parents=True, exist_ok=True)
                            fixed_processed_subject.label.save(patient_output_dir / 'CT_seg' / ct_label_path.name)
                        
                        if moving_processed_subject.label:
                            (patient_output_dir / 'MR_seg').mkdir(parents=True, exist_ok=True)
                            moving_processed_subject.label.save(patient_output_dir / 'MR_seg' / mr_label_path.name)
                        
                        print(f"    - ✓ Fallback processing successful for {patient_id}")
                        
                    except Exception as fallback_e:
                        print(f"    - ✗ Fallback failed for {patient_id}: {fallback_e}")
                        continue
                else:
                    print(f"    - ✗ Runtime error for {patient_id}: {e}")
                    continue
            except Exception as e:
                print(f"    - ✗ Unexpected error for {patient_id}: {e}")
                continue

    print("\nPreprocessing complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Comprehensive preprocessing script for MR-CT registration using TorchIO.")
    parser.add_argument("--dicom_root", type=str, required=True, help="Root directory of the original CHAOS DICOM files (containing Train/Test).")
    parser.add_argument("--png_root", type=str, required=True, help="Root directory of the original CHAOS PNG label files (containing Train/Test).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed files.")
    parser.add_argument("--liver_handling", type=str, default='analyze', 
                       choices=['analyze', 'intersection', 'crop', 'ignore'],
                       help="Strategy for handling mismatched liver labels: "
                            "analyze (report only), intersection (keep overlap), "
                            "crop (crop to common region), ignore (no modification)")
    parser.add_argument("--max-patients", type=int, default=None, help="Maximum number of patients to process (for testing)")
    args = parser.parse_args()
    
    preprocess_dataset(args.dicom_root, args.png_root, args.output_dir, 
                       liver_handling=args.liver_handling, max_patients=args.max_patients)