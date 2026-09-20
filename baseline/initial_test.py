import os
import numpy as np
import nibabel as nib
from medpy.metric import hd95
from scipy.ndimage import binary_erosion
import argparse
import csv
import glob
from pathlib import Path

def compute_dice_coefficient(mask1, mask2):
    """
    Compute the Dice Similarity Coefficient between two binary masks.
    """
    intersection = np.sum(mask1 * mask2)
    size1 = np.sum(mask1)
    size2 = np.sum(mask2)
    return (2.0 * intersection + 1e-5 ) / (size1 + size2 + 1e-5) 

def compute_label_wise_dice(seg1, seg2, labels):
    dice_results = []
    for label in labels:
        # Isolate current label in both segmentations
        seg1_label = seg1 == label
        seg2_label = seg2 == label

        # Compute 95% HD for the current label
        dice = compute_dice_coefficient(seg1_label, seg2_label)
        dice_results.append(dice)

    return dice_results

def extract_surface_points(seg):
    binary_seg = seg > 0
    eroded = binary_erosion(binary_seg)
    surface = binary_seg ^ eroded
    return np.array(np.where(surface)).T

def compute_95_hausdorff_distance(mask1, mask2):
    #seg1_points = extract_surface_points(seg1)
    #seg2_points = extract_surface_points(seg2)
    hd95_value = hd95(mask1.astype(bool), mask2.astype(bool))
    return hd95_value

def compute_label_wise_95hd(seg1, seg2, labels):
    hd95_results = []
    for label in labels:
        seg1_label = (seg1 == label)
        seg2_label = (seg2 == label)

        # Skip if either label is missing
        if not np.any(seg1_label) or not np.any(seg2_label):
            hd95_results.append(np.nan)  # or some sentinel value
            continue

        hd = compute_95_hausdorff_distance(seg1_label, seg2_label)
        hd95_results.append(hd)

    return hd95_results

def score_case(seg_fixed, seg_moving, label_list):
    dice_coefficient = compute_label_wise_dice(seg_fixed, seg_moving, label_list)
    dice_coefficient = np.array(dice_coefficient)

    hd95 = compute_label_wise_95hd(seg_fixed, seg_moving, label_list)
    hd95 = np.array(hd95)

    return {'DICE': dice_coefficient,
            'HD95': hd95}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Name of dataset (CHAOS, L2R, MRXFDG)")
    parser.add_argument("--csv_dir", type=str, default=None, help="Path to csv folder")
    parser.add_argument("--label_dir", type=str, default=None, help="Path to labels folder aka test folder")
    args = parser.parse_args()
    
    dice_list = []
    hd_list = []

    label_path = args.label_dir
    output_path = os.path.join(args.csv_dir, f"{args.data}_initial_metrics.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for patient in os.listdir(label_path):
        patient_path = os.path.join(label_path, patient)
        if os.path.isdir(patient_path):
            # Convert to Path object or use glob.glob
            patient_path_obj = Path(patient_path)
            
            # Find the files using pathlib
            mr_seg_files = list(patient_path_obj.glob('MR_seg/*.nii.gz'))
            ct_seg_files = list(patient_path_obj.glob('CT_seg/*.nii.gz'))
            
            # Check if files exist
            if not mr_seg_files or not ct_seg_files:
                print(f"Warning: Missing segmentation files for patient {patient}, skipping...")
                continue
                
            moving_path = mr_seg_files[0]
            fixed_path = ct_seg_files[0]
            
            try:
                seg_fixed = nib.load(fixed_path).get_fdata()
                seg_moving = nib.load(moving_path).get_fdata()

                # Compute both organ-level and overall metrics
                result = score_case(seg_fixed, seg_moving, [1, 2, 3, 4])
                dice_list.append(result['DICE'])
                hd_list.append(result['HD95'])
                #print(f"Patient {patient} Dice: {result['DICE']:.4f}, HD95: {result['HD95']:.4f}")
            except Exception as e:
                print(f"Error processing patient {patient}: {e}")
                continue

    dice_array = np.array(dice_list) 
    hd_array = np.array(hd_list)
    
    if len(dice_list) == 0:
        print("No valid patients processed. Check your data paths and file structure.")
        exit(1)
    
    print('DICE mean', np.mean(dice_array), 'DICE std', np.std(dice_array))
    print('DICE mean by organ', np.nanmean(dice_array, axis=0), 'DICE std', np.nanstd(dice_array, axis=0))
    print('HD95 mean', np.nanmean(hd_array), 'HD95 std', np.nanstd(hd_array))
    
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        if len(dice_list) > 0:
            writer.writerow(['DICE mean', np.mean(dice_array)])
            writer.writerow(['DICE std', np.std(dice_array)])
            writer.writerow(['DICE mean by organ', np.nanmean(dice_array, axis=0)])
            writer.writerow(['DICE std by organ', np.nanstd(dice_array, axis=0)])
            writer.writerow(['HD95 mean', np.nanmean(hd_array)])
            writer.writerow(['HD95 std', np.nanstd(hd_array)])
            print(f"Results saved to: {output_path}")
        else:
            writer.writerow(['Error', 'No valid patients processed'])
            print(f"Error log saved to: {output_path}")