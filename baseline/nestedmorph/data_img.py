import os
import shutil
import pickle
import nibabel as nib
import numpy as np

# Source and destination directories
source_dir = "../datasets/L2R_normalised"
destination_dir = "../datasets/L2R_voxelmorph"

# Create the destination structure
for split in ["Train", "Test"]:
    os.makedirs(os.path.join(destination_dir, split, "MR"), exist_ok=True)
    os.makedirs(os.path.join(destination_dir, split, "CT"), exist_ok=True)

# Iterate through Train and Test folders
for split in ["Train", "Test"]:
    split_path = os.path.join(source_dir, split)
    for patient_folder in os.listdir(split_path):
        patient_path = os.path.join(split_path, patient_folder)
        if os.path.isdir(patient_path):
            # Process MR and CT folders
            mr_path = os.path.join(patient_path, "MR")
            ct_path = os.path.join(patient_path, "CT")

            # MR to T1_moving
            if os.path.exists(mr_path):
                for mr_file in os.listdir(mr_path):
                    if mr_file.endswith(".nii.gz"):
                        mr_filepath = os.path.join(mr_path, mr_file)
                        with open(os.path.join(destination_dir, split, "MR", f"{patient_folder}_mr.pkl"), "wb") as f:
                            pickle.dump(nib.load(mr_filepath).get_fdata().astype(np.float32), f)

            # CT to Diffusion_Fixed
            if os.path.exists(ct_path):
                for ct_file in os.listdir(ct_path):
                    if ct_file.endswith(".nii.gz"):
                        ct_filepath = os.path.join(ct_path, ct_file)
                        with open(os.path.join(destination_dir, split, "CT", f"{patient_folder}_ct.pkl"), "wb") as f:
                            pickle.dump(nib.load(ct_filepath).get_fdata().astype(np.float32), f)
