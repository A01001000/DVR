import torchio as tio
import os
from glob import glob
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None, help="Path to train image folder")
    parser.add_argument("--aug_dir", type=str, default=None, help="Path to folder to save augmented images")
    parser.add_argument("--num_augs", type=int, default=10, help="Number of augmented versions for each original image")
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    os.makedirs(args.aug_dir, exist_ok=True)

    # augmentation pipeline
    transforms = tio.Compose([
        tio.RandomAffine(scales=(0.9, 1.2), degrees=10),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=7.5),
        tio.RandomNoise(p=0.5),
        tio.RandomBlur(p=0.5),
    ])

    patients = sorted(os.listdir(args.dir))
    for i, pid in enumerate(patients):
        mr_path = glob(os.path.join(args.dir, pid, "MR", "*.nii.gz"))[0]
        ct_path = glob(os.path.join(args.dir, pid, "CT", "*.nii.gz"))[0]
        
        subject = tio.Subject(
            mr=tio.ScalarImage(mr_path),
            ct=tio.ScalarImage(ct_path),
        )
    
        # Generate N augmented versions
        for j in range(args.num_augs):
            # Apply the random transform
            transformed_subject = transforms(subject)
        
            # Create a new directory for the augmented subject
            output_mr_dir = os.path.join(args.aug_dir, f"{pid}.{j}", "MR")
            output_ct_dir = os.path.join(args.aug_dir, f"{pid}.{j}", "CT")
            os.makedirs(output_mr_dir, exist_ok=True)
            os.makedirs(output_ct_dir, exist_ok=True)
        
            # Save the new augmented images
            transformed_subject.mr.save(os.path.join(output_mr_dir, f"subject_{i:03d}_aug_{j:02d}_mr.nii.gz"))
            transformed_subject.ct.save(os.path.join(output_ct_dir, f"subject_{i:03d}_aug_{j:02d}_ct.nii.gz"))

        print(f"Generated {args.num_augs} versions for subject {i+1}/{len(patients)}")