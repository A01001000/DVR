import torchio as tio
from pathlib import Path
import argparse
import os
import time
import torch
import tempfile
import nibabel as nib
import numpy as np
import cv2
import shutil
import subprocess

# Set TemplateFlow cache directory to writable location BEFORE importing 
# TODO: REMOVE
os.environ['TEMPLATEFLOW_HOME'] = '/templateflow_cache'

try:
    import ants
    ANTS_AVAILABLE = True
except ImportError:
    ANTS_AVAILABLE = False
    print("Warning: ANTs not available. Atlas registration will be skipped.")

try:
    from nilearn import datasets
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    print("Warning: Nilearn not available. Will try to install or skip atlas download.")

try:
    from templateflow import api as tf_api
    TEMPLATEFLOW_AVAILABLE = True
except ImportError:
    TEMPLATEFLOW_AVAILABLE = False
    print("Warning: TemplateFlow not available. Will use nilearn for atlas download.")


def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine, nii.header

def save_nifti(data, affine, header, path):
    nii = nib.Nifti1Image(data, affine, header)
    nib.save(nii, path)

def download_mni_atlas(atlas_dir):
    """
    Download MNI atlas and labels using nilearn.
    Returns paths to atlas and labels files.
    """
    atlas_path = Path(atlas_dir)
    atlas_path.mkdir(parents=True, exist_ok=True)
    
    atlas_file = atlas_path / 'atlas.nii.gz'
    labels_file = atlas_path / 'labels.nii.gz'
    
    # Check if files already exist
    if atlas_file.exists() and labels_file.exists():
        print(f"Atlas files already exist in {atlas_dir}")
        return str(atlas_file), str(labels_file)
    
    try:
        # Import nilearn datasets
        from nilearn import datasets
        print("Downloading MNI template and Harvard-Oxford atlas using nilearn...")
        
        # Download MNI152 template (1mm resolution)
        template_img = datasets.load_mni152_template(resolution=1)
        
        # Download Harvard-Oxford cortical atlas (thresholded max-probability labels)
        ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
        
        # Save template image
        template_img.to_filename(str(atlas_file))
        print(f"✓ Atlas template saved: {atlas_file}")
        
        # Save label image 
        import shutil
        shutil.copy(ho_atlas.maps, str(labels_file))
        print(f"✓ Atlas labels saved: {labels_file}")
        print(f"✓ Label names: {len(ho_atlas.labels)} regions")
        
        return str(atlas_file), str(labels_file)
        
    except ImportError:
        print("Nilearn not available. Attempting to install...")
        try:
            import subprocess
            subprocess.check_call(['pip', 'install', 'nilearn'])
            # Try importing again
            from nilearn import datasets
            print("Nilearn installed successfully! Downloading atlas...")
            
            # Download MNI152 template (1mm resolution)
            template_img = datasets.load_mni152_template(resolution=1)
            
            # Download Harvard-Oxford cortical atlas (thresholded max-probability labels)
            ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
            
            # Save template image
            template_img.to_filename(str(atlas_file))
            print(f"✓ Atlas template saved: {atlas_file}")
            
            # Save label image 
            import shutil
            shutil.copy(ho_atlas.maps, str(labels_file))
            print(f"✓ Atlas labels saved: {labels_file}")
            print(f"✓ Label names: {len(ho_atlas.labels)} regions")
            
            return str(atlas_file), str(labels_file)
            
        except Exception as install_e:
            print(f"Failed to install nilearn: {install_e}")
            return None, None
        
    except Exception as e:
        print(f"Failed to download atlas using nilearn: {e}")
        
        try:
            # Fallback: try to create a simple brain mask from any available template
            from nilearn import datasets
            template_img = datasets.load_mni152_template(resolution=1)
            
            # Save template
            template_img.to_filename(str(atlas_file))
            
            # Create simple brain mask from template
            data = template_img.get_fdata()
            
            # Create binary brain mask (threshold at 10% of max intensity)
            mask = (data > 0.1 * data.max()).astype(np.uint8)
            mask_img = nib.Nifti1Image(mask, template_img.affine, template_img.header)
            nib.save(mask_img, str(labels_file))
            
            print(f"✓ Created simple brain mask: {labels_file}")
            return str(atlas_file), str(labels_file)
            
        except Exception as e2:
            print(f"Complete atlas download failure: {e2}")
            return None, None

def register_atlas_to_image(image_path, atlas_image_path, atlas_labels_path, output_path):
    """
    Register MNI atlas to subject image and warp labels using ANTs Python library.
    """
    try:
        import ants
        
        print(f"    - Loading subject image: {image_path}")
        print(f"    - Loading atlas: {atlas_image_path}")
        print(f"    - Loading atlas labels: {atlas_labels_path}")
        
        # Load images directly with ANTs
        subject_img = ants.image_read(str(image_path))
        atlas_img = ants.image_read(str(atlas_image_path))
        atlas_labels = ants.image_read(str(atlas_labels_path))
        
        # Registration (atlas → subject)
        print("    - Performing SyN registration...")
        tx = ants.registration(
            fixed=subject_img, 
            moving=atlas_img, 
            type_of_transform="SyN",
            verbose=False
        )
        
        # Warp labels using NearestNeighbor interpolation
        print("    - Warping atlas labels to subject space...")
        warped_labels = ants.apply_transforms(
            fixed=subject_img,
            moving=atlas_labels,
            transformlist=tx['fwdtransforms'],
            interpolator='nearestNeighbor'
        )
        
        # Save output directly using ANTs image_write
        ants.image_write(warped_labels, str(output_path))
        print(f"    - Atlas labels saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"    - Atlas registration failed: {e}")
        print("    - Skipping atlas-based label generation")
        return False

def register_images_ants(fixed_subject, moving_subject):
    """
    Performs affine registration to align a moving subject to a fixed subject.
    Returns a new TorchIO Subject with the warped moving image.
    """
    print("    - Performing affine registration (MR -> CT)...")
    
    try:
        import ants
        
        # Convert TorchIO images to ANTs images
        # Get the numpy data and affine from TorchIO
        def to_numpy(arr):
            return arr.numpy() if hasattr(arr, "numpy") else arr

        fixed_data = to_numpy(fixed_subject.CT.data.squeeze())
        fixed_affine = to_numpy(fixed_subject.CT.affine)
        moving_data = to_numpy(moving_subject.MR.data.squeeze())
        moving_affine = to_numpy(moving_subject.MR.affine)
        
        # Create ANTs images
        fixed_img_ants = ants.from_numpy(
            fixed_data, 
            origin=tuple(fixed_affine[:3, 3]),
            spacing=tuple(np.abs(np.diag(fixed_affine[:3, :3])))
        )
        moving_img_ants = ants.from_numpy(
            moving_data,
            origin=tuple(moving_affine[:3, 3]), 
            spacing=tuple(np.abs(np.diag(moving_affine[:3, :3])))
        )

        # Perform affine (rigid + scaling) registration
        tx = ants.registration(
            fixed=fixed_img_ants,
            moving=moving_img_ants,
            type_of_transform='Affine',
            verbose=False
        )
        
        # Apply the transform to the moving image
        warped_moving_img_ants = ants.apply_transforms(
            fixed=fixed_img_ants,
            moving=moving_img_ants,
            transformlist=tx['fwdtransforms']
        )
        
        # Convert back to TorchIO
        warped_data = warped_moving_img_ants.numpy()
        warped_tensor = torch.from_numpy(warped_data).unsqueeze(0).float()
        
        # Create a new TorchIO subject from the warped data
        # Use the fixed image's affine since we registered to that space
        warped_subject = tio.Subject(
            MR=tio.ScalarImage(tensor=warped_tensor, affine=fixed_subject.CT.affine)
        )
        
        return warped_subject
        
    except Exception as e:
        print(f"    - ANTs registration failed: {e}")
        print("    - Returning original moving subject (no registration)")
        return moving_subject

def verify_output_properties(ct_subject, mr_subject, patient_id, target_spacing=(1, 1, 1), target_size=(192, 192, 192)):
    """
    Verify that the processed subjects have the correct spacing and size.
    """
    print(f"    - Verification for {patient_id}:")
    
    # Check CT properties
    ct_spacing = tuple(ct_subject.CT.spacing)
    ct_shape = tuple(ct_subject.CT.shape[1:])  # Remove channel dimension
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

def preprocess_dataset(root_dir, output_dir, atlas_dir=None, final_size=(192,192,192), target_spacing=(1,1,1), max_patients=None):
    """
    Main function to preprocess the entire dataset using only TorchIO.
    This pipeline performs initial alignment via resampling, modality-specific
    normalization, and final resizing, ensuring labels and images stay matched.
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load atlas if provided, or download if needed
    atlas_image_path = None
    atlas_labels_path = None
    
    if atlas_dir:
        atlas_path = Path(atlas_dir)
        
        # First, try to find existing atlas files
        possible_atlas_files = [
            'MNI152_T1_1mm.nii.gz',  # Downloaded via nilearn
            'mni_icbm152_t1_tal_nlin_asym_09a.nii.gz', 
            'atlas.nii.gz', 
            'template.nii.gz'
        ]
        possible_label_files = [
            'HarvardOxford-cort-maxprob-thr25-1mm.nii.gz',  # Downloaded via nilearn
            'mni_icbm152_CerebrA_tal_nlin_asym_09a.nii.gz', 
            'labels.nii.gz', 
            'atlas_labels.nii.gz',
            'HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz'  # In subfolder
        ]
        
        for atlas_file in possible_atlas_files:
            if (atlas_path / atlas_file).exists():
                atlas_image_path = str(atlas_path / atlas_file)
                break
        
        for label_file in possible_label_files:
            if (atlas_path / label_file).exists():
                atlas_labels_path = str(atlas_path / label_file)
                break
        
        # If no atlas found, try to download using TemplateFlow
        if not (atlas_image_path and atlas_labels_path):
            print("Atlas files not found locally. Attempting to download...")
            downloaded_atlas, downloaded_labels = download_mni_atlas(atlas_dir)
            if downloaded_atlas and downloaded_labels:
                atlas_image_path = downloaded_atlas
                atlas_labels_path = downloaded_labels
        
        if atlas_image_path and atlas_labels_path and ANTS_AVAILABLE:
            print(f"✓ Found atlas: {atlas_image_path}")
            print(f"✓ Found atlas labels: {atlas_labels_path}")
            print(f"✓ ANTs available: {ANTS_AVAILABLE}")
        else:
            print(f"✗ Atlas setup failed:")
            print(f"  - atlas_image_path: {atlas_image_path}")
            print(f"  - atlas_labels_path: {atlas_labels_path}")
            print(f"  - ANTS_AVAILABLE: {ANTS_AVAILABLE}")
            if not ANTS_AVAILABLE:
                print("  -> ANTs not available. Label generation will be skipped.")
            else:
                print("  -> Atlas files not found and download failed. Label generation will be skipped.")
            atlas_image_path = None
            atlas_labels_path = None
    
    # --- Step 1: Create Histogram Landmarks from the Training Set ---
    train_mr_files = list((root_path / 'Train').glob('*/MR/*.nii.gz'))
    landmarks_path = output_path / 'mr_landmarks.pt'
    
    print(f"Found {len(train_mr_files)} MR files in the Train set for landmark creation.")
    
    if landmarks_path.is_file():
        print(f"Loading existing landmarks from {landmarks_path}")
        landmarks_dict = torch.load(landmarks_path, weights_only=False)
        if not isinstance(landmarks_dict, dict):
            # If loaded object is not a dict, wrap it
            landmarks_dict = {'MR': landmarks_dict}
    elif train_mr_files:
        print(f"Landmarks file not found. Creating it...")
        landmarks_dict = create_histogram_landmarks(train_mr_files, landmarks_path)
    else:
        raise FileNotFoundError("Cannot create or load landmarks: No MR files found in the Train set.")

    # --- Step 2: Define the Preprocessing Transforms ---
    
    resample = tio.Resample(target_spacing)
    crop_pad = tio.CropOrPad(final_size)
    ct_strip_and_enhance = CTEnhancement(skull_dir, window_center=40, window_width=80)
    mr_strip = MRSkullStripANTs(skull_dir)
    mr_normalize = tio.HistogramStandardization(landmarks=landmarks_dict)

    # --- Step 3: Process Each Split (Train, Test) ---
    for split in ['Train', 'Test']:
        split_path = root_path / split
        if not split_path.is_dir():
            print(f"Split '{split}' not found, skipping.")
            continue
            
        print(f"\n--- Processing {split} set ---")
        
        patient_dirs = sorted([p for p in split_path.iterdir() if p.is_dir()])
        
        # Apply max_patients limit if specified
        if max_patients is not None:
            patient_dirs = patient_dirs[:max_patients]
            print(f"Processing only first {len(patient_dirs)} patients (max_patients={max_patients})")
        
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
                fixed_subject_dict = {'CT': tio.ScalarImage(ct_img_path)}
                if ct_label_path:
                    fixed_subject_dict['label'] = tio.LabelMap(ct_label_path)
                fixed_subject = tio.Subject(fixed_subject_dict)
                
                moving_subject_dict = {'MR': tio.ScalarImage(mr_img_path)}
                if mr_label_path:
                    moving_subject_dict['label'] = tio.LabelMap(mr_label_path)
                moving_subject = tio.Subject(moving_subject_dict)
                
                print(f"    - Loaded subjects: CT has {len(fixed_subject_dict)} items, MR has {len(moving_subject_dict)} items")
                
                # Debug: Check if subjects are valid before processing
                print(f"    - Debug: Fixed subject keys: {list(fixed_subject.keys())}")
                print(f"    - Debug: Moving subject keys: {list(moving_subject.keys())}")
                
                # --- Apply the Proven Processing Pipeline ---
                print("    - Applying proven pipeline: resample → crop → resize → enhance...")
                start_time = time.time()

                # Resample both to same spacing first
                print("    - Resampling to target spacing...")
                ct_resampled = resample(fixed_subject)
                mr_resampled = resample(moving_subject)
                
                # Skull strip and crop each image independently
                print("    - Skull stripping with ants...")
                ct_cropped = ct_strip_and_enhance(ct_resampled)
                mr_cropped = mr_strip(mr_resampled)
                
                # Affine Registration: Align the cropped MR to the cropped CT
                print("    - Affine registration...")
                mr_registered = register_images_ants(fixed_subject=ct_cropped, moving_subject=mr_cropped)
                
                # Intensity Normalization on the *registered* MR
                print("    - Intensity Normalization on the *registered* MR...")
                mr_normalized = mr_normalize(mr_registered)
                
                # Final Resizing (Padding)
                print("    - Final Resizing (Padding)...")
                fixed_processed_subject = crop_pad(ct_cropped)
                moving_processed_subject = crop_pad(mr_normalized)
                
                print(f"    - Processing done in {time.time() - start_time:.2f}s")
                
                # --- Verify output properties ---
                verify_output_properties(fixed_processed_subject, moving_processed_subject, patient_id, 
                                       target_spacing=target_spacing, target_size=final_size)
                
                # --- Save the preprocessed data ---
                patient_output_dir = output_path / split / patient_id
                
                # Save images
                (patient_output_dir / 'CT').mkdir(parents=True, exist_ok=True)
                (patient_output_dir / 'MR').mkdir(parents=True, exist_ok=True)
                ct_output_path = patient_output_dir / 'CT' / ct_img_path.name
                mr_output_path = patient_output_dir / 'MR' / mr_img_path.name
                fixed_processed_subject.CT.save(ct_output_path)
                moving_processed_subject.MR.save(mr_output_path)

                # Only generate/save labels for the Test set
                if split == 'Test':
                    # Generate atlas-based labels if atlas is available
                    if atlas_image_path and atlas_labels_path:
                        print("    - Generating atlas-based labels for Test set...")
                        
                        # Create label directories
                        (patient_output_dir / 'CT_seg').mkdir(parents=True, exist_ok=True)
                        (patient_output_dir / 'MR_seg').mkdir(parents=True, exist_ok=True)
                        
                        # Register atlas to CT and generate labels
                        ct_labels_path = patient_output_dir / 'CT_seg' / 'brain_labels.nii.gz'
                        register_atlas_to_image(
                            str(ct_output_path),
                            atlas_image_path,
                            atlas_labels_path,
                            str(ct_labels_path)
                        )
                        
                        # Register atlas to MR and generate labels
                        mr_labels_path = patient_output_dir / 'MR_seg' / 'brain_labels.nii.gz'
                        register_atlas_to_image(
                            str(mr_output_path),
                            atlas_image_path,
                            atlas_labels_path,
                            str(mr_labels_path)
                        )
                    
                    (patient_output_dir / 'CT_seg').mkdir(parents=True, exist_ok=True)
                    if ct_label_path is not None and hasattr(fixed_processed_subject, 'label'):
                        fixed_processed_subject.label.save(patient_output_dir / 'CT_seg' / ct_label_path.name)
                    
                    (patient_output_dir / 'MR_seg').mkdir(parents=True, exist_ok=True)
                    if mr_label_path is not None and hasattr(moving_processed_subject, 'label'):
                        moving_processed_subject.label.save(patient_output_dir / 'MR_seg' / mr_label_path.name)
                
                # For the Train set, ensure no segmentation folders exist
                elif split == 'Train':
                    for seg_dir in ['CT_seg', 'MR_seg']:
                        seg_path = patient_output_dir / seg_dir
                        if seg_path.exists():
                            print(f"    - Removing existing segmentation folder for Train set: {seg_path}")
                            shutil.rmtree(seg_path)

            except Exception as e:
                print(f"    - Skipping patient {patient_id}. Reason: {e}")
                continue

    print("\nPreprocessing complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Comprehensive preprocessing script for MR-CT registration using TorchIO.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory of the dataset (containing Train/Test folders).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed files.")
    parser.add_argument("--atlas_dir", type=str, default="./atlases", help="Directory to store/find MNI atlas files. Will download if not found.")
    parser.add_argument("--skull_dir", type=str, default="./skull", help="Directory to store/find MNI atlas files. Will download if not found.")
    parser.add_argument("--ants_path", type=str, default=None, help="Path to the ANTs binary.")
    parser.add_argument("--max_patients", type=int, default=None, help="Maximum number of patients to process (for testing)")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[1,1,1], help="Target spacing for resampling (e.g., 1 1 1).")
    parser.add_argument("--final_size", type=int, nargs=3, default=[192,192,192], help="Final size for all volumes (e.g., 192 192 192).")
    args = parser.parse_args()
    
    validated_ants_script_path = None # Use a new, unambiguous variable name
    if args.ants_path:
        ants_script_path_obj = Path(args.ants_path) / 'antsBrainExtraction.sh'
        
        if ants_script_path_obj.is_file():
            print(f"✅ ANTs script found: {ants_script_path_obj}")
            validated_ants_script_path = str(ants_script_path_obj) # Assign the full script path
        else:
            print(f"❌ FATAL ERROR: 'antsBrainExtraction.sh' not found in the directory: {args.ants_path}")
            exit(1)
    else:
        print("⚠️ Warning: --ants_path not provided. Skull stripping will use the fallback method.")
        
    preprocess_dataset(args.input_dir, args.output_dir, args.atlas_dir, args.skull_dir, validated_ants_script_path,
                      final_size=tuple(args.final_size), 
                      target_spacing=tuple(args.target_spacing), 
                      max_patients=args.max_patients)