
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
import json
from os import path
# from utils.foerstner import foerstner_kpts
# from utils.vxmplusplus_utils import MINDSSC


def get_files(data_dir, task, mode):

    if task == "ThoraxCBCT" or task == "OncoReg":
        data_json = os.path.join(data_dir, task + "_dataset.json")
        with open(data_json) as file:
            data = json.load(file)

        if mode == 'Tr':
            mode1 = 'training_paired_images'
        elif mode == 'Val':
            mode = 'Tr'
            mode1 = 'registration_val'
        elif mode == 'Ts':
            mode1 = 'registration_test'

        img_fixed_all = []
        img_moving_all = []
        kpts_fixed_all = []
        kpts_moving_all = []
        orig_shapes_all = []
        mind_fixed_all = []
        mind_moving_all = []
        case_list = []
        keypts_fixed_all = []
        img_mov_unmasked = []
        aff_mov_all = []
        mask_fixed_all = []
        mask_moving_all = []

        for pair in data[mode1]:
            print('loading data', pair["fixed"], pair["moving"])
            nam_fixed = os.path.basename(pair["fixed"]).split(".")[0]
            nam_moving = os.path.basename(pair["moving"]).split(".")[0]
            if nam_fixed.split('.nii.gz')[0].split('_')[2]=='0001':
                kpts_dir = path.join(data_dir, 'keypoints01')
            else:
                kpts_dir = path.join(data_dir, 'keypoints02')
     
            case_list.append(nam_fixed)

            img_fixed = nib.load(os.path.join(data_dir, "images" + mode, nam_fixed + ".nii.gz")).get_fdata()
            img_moving = nib.load(os.path.join(data_dir, "images" + mode, nam_moving + ".nii.gz")).get_fdata()
            aff_mov = nib.load(os.path.join(data_dir, "images" + mode, nam_moving + ".nii.gz")).affine


            kpts_fixed = np.loadtxt(os.path.join(kpts_dir + mode, nam_fixed + ".csv"),delimiter=',')
            kpts_moving = np.loadtxt(os.path.join(kpts_dir + mode, nam_moving + ".csv"),delimiter=',')


            mask_fixed = nib.load(os.path.join(data_dir, 'masks' + mode, nam_fixed + ".nii.gz")).get_fdata()
            mask_moving = nib.load(os.path.join(data_dir, 'masks' + mode, nam_moving + ".nii.gz")).get_fdata()
            # masked_fixed = F.interpolate(((img_fixed+1024)*label_fixed).unsqueeze(0).unsqueeze(0),scale_factor=.5,mode='trilinear').squeeze()
            # masked_moving = F.interpolate(((img_moving+1024)*label_moving).unsqueeze(0).unsqueeze(0),scale_factor=.5,mode='trilinear').squeeze()


            # shape = label_fixed.shape
            # img_fixed_all.append(masked_fixed)
            # img_moving_all.append(masked_moving)

            kpts_fixed_all.append(kpts_fixed)
            kpts_moving_all.append(kpts_moving)
            orig_shapes_all.append(img_moving.shape)
            img_moving_all.append(img_moving)
            img_fixed_all.append(img_fixed)
            aff_mov_all.append(aff_mov)

            mask_fixed_all.append(mask_fixed)
            mask_moving_all.append(mask_moving)
            #break
    else:
        raise ValueError(f"Task {task} undefined!")
    
    return img_fixed_all, img_moving_all, kpts_fixed_all, kpts_moving_all, orig_shapes_all, case_list, aff_mov_all, mask_fixed_all, mask_moving_all


def get_files_mrct(data_dir, task, mode):

    task = 'AbdomenMRCT'

    data_json = os.path.join(data_dir, task + "_dataset_edit.json")
    with open(data_json) as file:
        data = json.load(file)

    if mode == 'Tr':
        mode1 = 'training_paired_images'
    elif mode == 'Val':
        mode1 = 'test_paired_images'

    img_fixed_all = []
    img_moving_all = []

    case_list = []

    aff_mov_all = []
    mask_fixed_all = []
    mask_moving_all = []
    seg_fixed_all = []
    seg_moving_all = []

    for pair in data[mode1]:
        # print('loading data', pair["fixed"], pair["moving"])
        print('loading data', pair["0"], pair["1"])
        nam_fixed = os.path.basename(pair["0"]).split(".")[0]
        nam_moving = os.path.basename(pair["1"]).split(".")[0]

        #0 is MR and should be fixed image


        case_list.append(nam_fixed)

        img_fixed = nib.load(os.path.join(data_dir, "images" + mode, nam_fixed + ".nii.gz")).get_fdata()
        img_moving = nib.load(os.path.join(data_dir, "images" + mode, nam_moving + ".nii.gz")).get_fdata()
        aff_mov = nib.load(os.path.join(data_dir, "images" + mode, nam_moving + ".nii.gz")).affine

        mask_fixed = nib.load(os.path.join(data_dir, 'masks' + mode, nam_fixed + ".nii.gz")).get_fdata()
        mask_moving = nib.load(os.path.join(data_dir, 'masks' + mode, nam_moving + ".nii.gz")).get_fdata()

        if mode == 'Tr':
            seg_fixed = nib.load(os.path.join(data_dir, "labels" + mode, nam_fixed + ".nii.gz")).get_fdata()
            seg_moving = nib.load(os.path.join(data_dir, "labels" + mode, nam_moving + ".nii.gz")).get_fdata()
            seg_fixed_all.append(seg_fixed)
            seg_moving_all.append(seg_moving)

        img_moving_all.append(img_moving)
        img_fixed_all.append(img_fixed)
        aff_mov_all.append(aff_mov)

        mask_fixed_all.append(mask_fixed)
        mask_moving_all.append(mask_moving)


    # return img_fixed_all, img_moving_all, case_list, aff_mov_all, mask_fixed_all, mask_moving_all
    return img_fixed_all, img_moving_all, case_list, aff_mov_all, seg_fixed_all, seg_moving_all, mask_fixed_all, mask_moving_all
