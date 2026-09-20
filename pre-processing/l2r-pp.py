import torchio as tio
from pathlib import Path
import argparse
import os
import time
import torch

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

def preprocess_dataset(root_dir, output_dir, final_size=(192, 192, 192), target_spacing=(2, 2, 2)):
    """
    Main function to preprocess the entire dataset using only TorchIO.
    This pipeline performs initial alignment via resampling, modality-specific
    normalization, and final resizing, ensuring labels and images stay matched.
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # --- Step 1: Create Histogram Landmarks from the Training Set ---
    train_mr_files = list((root_path / 'Train').glob('*/MR/*.nii.gz'))
    landmarks_path = output_path / 'mr_landmarks.pt'
    
    print(f"Found {len(train_mr_files)} MR files in the Train set for landmark creation.")
    
    if landmarks_path.is_file():
        print(f"Loading existing landmarks from {landmarks_path}")
        landmarks_dict = torch.load(landmarks_path)
        if not isinstance(landmarks_dict, dict):
            # If loaded object is not a dict, wrap it
            landmarks_dict = {'MR': landmarks_dict}
    elif train_mr_files:
        print(f"Landmarks file not found. Creating it...")
        landmarks_dict = create_histogram_landmarks(train_mr_files, landmarks_path)
    else:
        raise FileNotFoundError("Cannot create or load landmarks: No MR files found in the Train set.")

    # --- Step 2: Define the Preprocessing Transforms ---
    
    # For CT: Clamp HU values, then rescale to 0-1
    ct_normalization = tio.Compose([
        tio.Clamp(out_min=-200, out_max=300),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])
    
    # For MR: Use histogram standardization based on the training set
    mr_normalization = tio.HistogramStandardization(landmarks=landmarks_dict)

    # Define spatial transforms - CRITICAL: Use CropOrPad to ensure exact size AND spacing
    # This guarantees exactly 192x192x192 at exactly 2x2x2mm spacing
    spatial_transforms = tio.Compose([
        tio.Resample(target_spacing),      # First ensure 2x2x2mm spacing
        tio.CropOrPad(final_size),         # Then force exactly 192x192x192 size
    ])
    
    # Define affine registration transform
    # This will register moving (MR) to fixed (CT) using mutual information
    affine_registration = tio.ToCanonical()  # Ensure canonical orientation first

    # --- Step 3: Process Each Split (Train, Test) ---
    for split in ['Train', 'Test']:
        split_path = root_path / split
        if not split_path.is_dir():
            print(f"Split '{split}' not found, skipping.")
            continue
            
        print(f"\n--- Processing {split} set ---")
        
        patient_dirs = sorted([p for p in split_path.iterdir() if p.is_dir()])
        
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            print(f"  -> Patient: {patient_id}")
            
            try:
                # Find the image and label files robustly
                ct_img_path = next(patient_dir.glob('CT/*.nii.gz'))
                mr_img_path = next(patient_dir.glob('MR/*.nii.gz'))
                
                # Use .glob which returns an empty list if not found, preventing errors
                ct_label_paths = list(patient_dir.glob('CT_seg/*.nii.gz'))
                mr_label_paths = list(patient_dir.glob('MR_seg/*.nii.gz'))
                ct_label_path = ct_label_paths[0] if ct_label_paths else None
                mr_label_path = mr_label_paths[0] if mr_label_paths else None

                # --- Load data into TorchIO Subjects ---
                # A Subject bundles an image with its label. This is the key to
                # guaranteeing that all transforms are applied to both identically.
                fixed_subject = tio.Subject(
                    image=tio.ScalarImage(ct_img_path),
                    label=tio.LabelMap(ct_label_path) if ct_label_path else None
                )
                
                moving_subject = tio.Subject(
                    MR=tio.ScalarImage(mr_img_path),
                    label=tio.LabelMap(mr_label_path) if mr_label_path else None
                )
                
                # --- Apply the Full Pipeline ---
                print("    - Normalizing and resizing...")
                start_time = time.time()

                # 1. First normalize intensities
                # Apply normalization to the entire subjects
                fixed_normalized_subject = tio.Subject(
                    image=ct_normalization(fixed_subject.image),
                    label=fixed_subject.label
                )
                moving_normalized_subject = mr_normalization(moving_subject)
                
                # 2. Apply spatial transforms to both (resize to 192³ and force 2x2x2mm spacing)
                fixed_processed_subject = spatial_transforms(fixed_normalized_subject)
                moving_processed_subject = spatial_transforms(moving_normalized_subject)
                
                # Both are now at exactly 2x2x2mm spacing and 192x192x192 size

                print(f"    - Processing done in {time.time() - start_time:.2f}s")
                
                # --- Verify output properties ---
                verify_output_properties(fixed_processed_subject, moving_processed_subject, patient_id)
                
                # --- Save the preprocessed data ---
                patient_output_dir = output_path / split / patient_id
                
                # Save images
                (patient_output_dir / 'CT').mkdir(parents=True, exist_ok=True)
                (patient_output_dir / 'MR').mkdir(parents=True, exist_ok=True)
                fixed_processed_subject.image.save(patient_output_dir / 'CT' / ct_img_path.name)
                moving_processed_subject.MR.save(patient_output_dir / 'MR' / mr_img_path.name)
                
                # Save labels
                if fixed_processed_subject.label:
                    (patient_output_dir / 'CT_seg').mkdir(parents=True, exist_ok=True)
                    fixed_processed_subject.label.save(patient_output_dir / 'CT_seg' / ct_label_path.name)
                
                if moving_processed_subject.label:
                    (patient_output_dir / 'MR_seg').mkdir(parents=True, exist_ok=True)
                    moving_processed_subject.label.save(patient_output_dir / 'MR_seg' / mr_label_path.name)

            except Exception as e:
                print(f"    - Skipping patient {patient_id}. Reason: {e}")
                continue

    print("\nPreprocessing complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Comprehensive preprocessing script for MR-CT registration using TorchIO.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory of the dataset (containing Train/Test folders).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed files.")
    args = parser.parse_args()
    
    preprocess_dataset(args.input_dir, args.output_dir)