import glob
import sys, os
import numpy as np
from sympy import use
import torch
from scipy.ndimage import zoom
from dinov2.eval.setup import build_model_for_eval
from dinov2.configs import load_and_merge_config
import torchvision.transforms as tt
import nibabel as nib
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from skimage.measure import label, regionprops
from skimage import morphology
import time
import torch.nn.functional as F
from scipy.ndimage import map_coordinates
import argparse

import scipy.ndimage
from scipy.ndimage import map_coordinates

from convex_adam_utils import *
import sys
import einops
from utils.img_operations import remove_uniform_intensity_slices, reconstruct_image, to_lungCT_window, clip_and_normalize_image, pca_lowrank_transform, MR_normalize
from utils.img_operations import extract_lung_mask
from utils.convexAdam_3D import convex_adam_3d, convex_adam_3d_w0, convex_adam_3d_interSmooth, convex_adam_3d_param, convex_adam_3d_param_dataSmooth
from utils.data_utils import get_files_mrct
from scipy.ndimage import laplace, gaussian_filter
import utils.img_operations as img_op
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist
from os import path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import csv
import pandas as pd

def save_middle_slice_png(volume, out_path, cmap='gray', vmin=None, vmax=None):
    mid = volume.shape[2] // 2
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(volume[:,:,mid], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_overlay_png(image, seg, out_path, label_cmap='jet', alpha=0.4, vmin=None, vmax=None):
    mid = image.shape[2] // 2
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(image[:,:,mid], cmap='gray', vmin=vmin, vmax=vmax)
    seg_slice = seg[:,:,mid]
    if np.max(seg_slice) > 0:
        plt.imshow(seg_slice, cmap=label_cmap, alpha=alpha, vmin=0, vmax=np.max(seg_slice))
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_overlay_png_masked(image, seg, out_path, label_cmap='jet', alpha=0.5, vmin=None, vmax=None):
    mid = image.shape[2] // 2
    img_slice = image[:,:,mid]
    seg_slice = seg[:,:,mid]
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax)
    mask = seg_slice > 0
    if np.any(mask):
        colored = np.zeros((*seg_slice.shape, 4))
        colored[mask] = plt.cm.get_cmap(label_cmap)(seg_slice[mask]/np.max(seg_slice[mask]))
        plt.imshow(colored, alpha=alpha)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_deformation_field_png(def_field, out_path):
    # def_field shape: (3, H, W, D) or (H, W, D, 3)
    if def_field.shape[0] == 3:
        # (3, H, W, D) -> (H, W, D, 3)
        def_field = np.moveaxis(def_field, 0, -1)
    if def_field.shape[-1] == 3:
        mid = def_field.shape[2] // 2
        u = def_field[:,:,mid,0]
        v = def_field[:,:,mid,1]
        mag = np.sqrt(u**2 + v**2)
        ang = np.arctan2(v, u)
        ang_norm = (ang + np.pi) / (2 * np.pi)
        mag_norm = mag / (np.max(mag) + 1e-8)
        hsv = np.zeros(u.shape + (3,), dtype=np.float32)
        hsv[...,0] = ang_norm
        hsv[...,1] = 1
        hsv[...,2] = mag_norm
        rgb = mcolors.hsv_to_rgb(hsv)
        plt.figure(figsize=(5,5))
        plt.axis('off')
        plt.imshow(rgb)
        plt.title('Deformation Field (middle slice)')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        # fallback to quiver for 2D
        mid = def_field.shape[2] // 2
        u = def_field[:,:,mid,0]
        v = def_field[:,:,mid,1]
        plt.figure(figsize=(5,5))
        plt.axis('off')
        plt.quiver(u, v)
        plt.title('Deformation Field (middle slice)')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def save_error_map_png(seg1, seg2, out_path):
    error_map = np.abs(seg1 - seg2)
    mid = error_map.shape[2] // 2
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(np.ones_like(error_map[:,:,mid]), cmap='gray', vmin=0, vmax=1)
    if np.max(error_map[:,:,mid]) > 0:
        plt.imshow(error_map[:,:,mid], cmap='hot', alpha=0.8, vmin=0, vmax=np.max(error_map[:,:,mid]))
    plt.title('Error Map (middle slice)')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
"""
FILE NOTE:

"""

class dinoReg:

    def __init__(self, dir, device_id=0, lr=1, smooth_weight=10, num_iter=1000, feat_size=(80,80)):
        self.device_id = device_id
        self.patch_size = 16
        # self.transform = tt.Compose([tt.Normalize(mean=0.5, std=0.2)])
        self.transform = tt.Compose([])
        self.patch_grid_size = 32
        self.patch_margin = 10
        self.src_slice_num = 2
        self.patch_grid_h, self.patch_grid_w = 8, 8
        self.slice_step = 5
        # self.img_size = (int(256 / self.patch_size+0.5) * self.patch_size * 3, int(192 / self.patch_size+0.5) * self.patch_size * 3)  #
        # self.img_size = (2080, 2560)  #set size
        # self.img_size = (1120, 1120)  #set size
        self.embed_dim = 384
        self.model = self.load_model(dir)
        self.img_size = (self.patch_size*feat_size[0], self.patch_size*feat_size[1])  #set size
        self.num_iter = num_iter


        self.batch_size = 12 #todo: implement parallel?
        # self.reg_featureDim = 1
        self.reg_featureDim = 24
        self.lr = lr
        self.smooth_weight = smooth_weight
        print('learning rate', self.lr)

        self.feature_height = self.img_size[0] // self.patch_size
        self.feature_width = self.img_size[1] // self.patch_size


    def extract_dinov2_feature(self, input_array):

        assert len(input_array.shape) == 3  # 2D image

        """flipping the input if needed"""
        # input_array = np.swapaxes(input_array, 0,1)

        input_rgb_array = input_array[np.newaxis, :, :, :]

        input_tensor = torch.Tensor(np.transpose(input_rgb_array, [0, 3, 1, 2]))
        input_tensor = self.transform(input_tensor)
        feature_array = self.model.forward_features(input_tensor.to(device=torch.device('cuda', self.device_id)))[
            'x_norm_patchtokens'].detach().cpu().numpy()
        del input_tensor

        return feature_array


    def case_inference(self, mov_arr, fix_arr, orig_img_shape, aff_mov,
                       mask_fixed=None, mask_moving=None, case_id='noID', disp_init=None, grid_sp_adam=1,DINOReg_useMask=True):

        assert len(mov_arr.shape) == 3

        """prepcocessing and feature extraction"""
        # mov_arr, fix_arr, slices_to_keep_indices, orig_chunked_shape, mask_fixed_arr, mask_moving_arr = self.case_preprocess(mov_arr, fix_arr, mask_fixed, mask_moving)
        mov_arr, fix_arr, slices_to_keep_indices, orig_chunked_shape, mask_fixed_arr, mask_moving_arr = self.case_preprocess(mov_arr, fix_arr)


        print('preprocessed moving and fixed image, shape', mov_arr.shape, fix_arr.shape)
        gap = 3 #3

        mov_feature = self.encode_3D_gap(mov_arr, gap=gap)
        print('encoded moving image')
        fix_feature = self.encode_3D_gap(fix_arr, gap=gap)
        print('encoded fixed image')

        feat_sliceNum = self.slice_num


        """PCA reduce dimension"""
        #only features inside the mask
        if DINOReg_useMask:
            # reshape to model output

            mask_fixed_arr = resize(mask_fixed_arr, (self.feature_height, self.feature_width, feat_sliceNum),
                                anti_aliasing=True)
            mask_moving_arr = resize(mask_moving_arr, (self.feature_height, self.feature_width, feat_sliceNum),
                                 anti_aliasing=True)
            mask_fixed_arr = np.where(mask_fixed_arr > 0.99, 1.0, 0)
            mask_moving_arr = np.where(mask_moving_arr > 0.99, 1.0, 0)
            # fixImg_1dim_threshold = nib.Nifti1Image(mask_fixed_arr, aff_mov)
            # nib.save(fixImg_1dim_threshold, os.path.join(output_dir, 'vis',case_list[i] + '_threshold.nii.gz'))

            # print('mask shape', mask_moving_arr.shape, mask_fixed_arr.shape)
            # print('feature  shape', mov_feature.shape, fix_feature.shape)
            mask_moving_arr = mask_moving_arr.flatten().astype(bool)
            mask_fixed_arr = mask_fixed_arr.flatten().astype(bool)
            mov_feature = mov_feature[mask_moving_arr, :]
            fix_feature = fix_feature[mask_fixed_arr, :]


        print('Starting PCA to reduce dimension')
        all_features = np.concatenate([mov_feature,fix_feature], axis=0)
        print('all features shape', all_features.shape, 'mask sum', mask_moving_arr.sum(), mask_fixed_arr.sum())
        pca_start_time = time.time()
        # object_pca = PCA(n_components=self.reg_featureDim) #what is SVD solver?
        # reduced_patches = object_pca.fit_transform(all_features)
        if configs['useSavedPCA']:
            reduced_patches = np.dot(all_features, PCA_matrix)
            eigenvalues = np.zeros(24)
        else:
            reduced_patches, eigenvalues = pca_lowrank_transform(all_features, self.reg_featureDim)
        print('PCA finished in {}, splitting features'.format(time.time()-pca_start_time))

        if DINOReg_useMask:
            mov_pca = np.zeros((self.feature_height * self.feature_width * feat_sliceNum, self.reg_featureDim), dtype='float32')
            fix_pca = np.zeros((self.feature_height * self.feature_width * feat_sliceNum, self.reg_featureDim), dtype='float32')
            mov_pca[mask_moving_arr, :] = reduced_patches[:mask_moving_arr.sum(), :]
            fix_pca[mask_fixed_arr, :] = reduced_patches[mask_moving_arr.sum():, :]
            mov_pca = mov_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])
            fix_pca = fix_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])
        else:

            mov_pca = reduced_patches[:feat_sliceNum * self.feature_height * self.feature_width, :]
            fix_pca = reduced_patches[feat_sliceNum * self.feature_height * self.feature_width:, :]
            mov_pca = mov_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])
            fix_pca = fix_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])

        eigenvalue_array.append(eigenvalues[:24])




        print('reshaping to original image shape')
        mov_pca_rescaled = resize(mov_pca, (orig_chunked_shape[0], orig_chunked_shape[1], orig_chunked_shape[2], self.reg_featureDim),
                                   anti_aliasing=True)
        fix_pca_rescaled = resize(fix_pca, (orig_chunked_shape[0], orig_chunked_shape[1], orig_chunked_shape[2], self.reg_featureDim),
                                   anti_aliasing=True)


        #plug in the slices to keep, the rest are 0
        mov_fullImg_pca_rescaled = np.zeros((orig_img_shape[0], orig_img_shape[1], orig_img_shape[2], self.reg_featureDim),
                                          dtype='float32')
        fix_fullImg_pca_rescaled = np.zeros((orig_img_shape[0], orig_img_shape[1], orig_img_shape[2], self.reg_featureDim),
                                          dtype='float32')

        mov_fullImg_pca_rescaled[:, :, slices_to_keep_indices, :] = mov_pca_rescaled
        fix_fullImg_pca_rescaled[:, :, slices_to_keep_indices, :] = fix_pca_rescaled

        """save copy of 1 channel feature for vis"""
        # for channel in range(3):
        #     mov_feat_1dim = mov_fullImg_pca_rescaled[:,:,:,channel:channel+3]
        #     fix_feat_1dim = fix_fullImg_pca_rescaled[:,:,:,channel:channel+3]
        #     movImg_1dim = nib.Nifti1Image(mov_feat_1dim, aff_mov)
        #     fixImg_1dim = nib.Nifti1Image(fix_feat_1dim, aff_mov)
        #     os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
        #     nib.save(movImg_1dim, os.path.join(output_dir, 'vis', case_id + '_mov_{}.nii.gz'.format(channel)))
        #     nib.save(fixImg_1dim, os.path.join(output_dir, 'vis', case_id + '_fix_{}.nii.gz'.format(channel)))
        #
        # sys.exit()

        # mov_feat_1dim = mov_fullImg_pca_rescaled[:,:,:,:3]
        # fix_feat_1dim = fix_fullImg_pca_rescaled[:,:,:,:3]
        # movImg_1dim = nib.Nifti1Image(mov_feat_1dim, aff_mov)
        # fixImg_1dim = nib.Nifti1Image(fix_feat_1dim, aff_mov)
        # nib.save(movImg_1dim, os.path.join(output_dir, 'vis' + case_list[i] + '_mov_feat_24dim.nii.gz'))
        # nib.save(fixImg_1dim, os.path.join(output_dir, 'vis' + case_list[i] + '_fix_feat_24dim.nii.gz'))
        # sys.exit()

        if save_feature:
            os.makedirs(os.path.join(output_dir_0, 'features'), exist_ok=True)
            np.save(os.path.join(output_dir_0, 'features', case_id + '_mov_feat.npy'), mov_fullImg_pca_rescaled)
            np.save(os.path.join(output_dir_0, 'features', case_id + '_fix_feat.npy'), fix_fullImg_pca_rescaled)

        """ConvexAdam optimization"""
        print('starting ConvexAdam optimization')

        # disp = convex_adam_3d(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled,
        # disp = convex_adam_3d_interSmooth(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled, #default 1000 iter
        #                       loss_func = "SSD", selected_niter=self.num_iter, lr=self.lr, selected_smooth=20, ic=True, lambda_weight=self.smooth_weight, disp_init=disp_init)
                              # loss_func = "SSD", selected_niter=5000, lr=self.lr, selected_smooth=3, ic=True, lambda_weight=self.smooth_weight, disp_init=disp_init)
                              # loss_func = "SSD", selected_niter=1000, lr=1, selected_smooth=3, ic=True, lambda_weight=10, disp_init=disp_init)

        disp = convex_adam_3d_param(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled, loss_func = "SSD", grid_sp_adam=grid_sp_adam,
                                               lambda_weight=configs['smooth_weight'], selected_niter=configs['num_iter'], lr=configs['lr'], disp_init=disp_init,
                                                iter_smooth_kernel = configs['iter_smooth_kernel'],
                                                iter_smooth_num = configs['iter_smooth_num'], end_smooth_kernel=1,final_upsample=configs['final_upsample'])
        

        """apply displacement field to moving image or landmarks"""

        """save copy of 1 channel feature for vis"""
        # mov_feat_1dim = mov_fullImg_pca_rescaled[:,:,:,:3]
        # fix_feat_1dim = fix_fullImg_pca_rescaled[:,:,:,:3]
        # movImg_1dim = nib.Nifti1Image(mov_feat_1dim, aff_mov)
        # fixImg_1dim = nib.Nifti1Image(fix_feat_1dim, aff_mov)
        # os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
        # nib.save(movImg_1dim, os.path.join(output_dir, 'vis', case_id + '_mov_feat_gap3.nii.gz'))
        # nib.save(fixImg_1dim, os.path.join(output_dir, 'vis',case_id + '_fix_feat_gap3.nii.gz'))
        # sys.exit()

        return disp

    def case_preprocess(self, mov_arr, fix_arr):
        assert len(mov_arr.shape) == 3
        assert len(fix_arr.shape) == 3

        pad_indices = []
        filtered_image_data, slices_to_keep_indices = remove_uniform_intensity_slices(fix_arr)
        pad_indices.append(slices_to_keep_indices)
        fix_arr = filtered_image_data
        mov_arr = mov_arr[:, :, slices_to_keep_indices]

        orig_chunked_shape = fix_arr.shape


        #old preprop
        fix_arr = MR_normalize(fix_arr)
        mov_arr = to_lungCT_window(mov_arr, wl=50, ww=400)
        mask_fixed = np.where(fix_arr > 0.05, 1.0, 0)
        mask_moving = np.where(mov_arr > 0.005, 1.0, 0)


        filtered_z = fix_arr.shape[2]


        mask_fixed = np.zeros_like(fix_arr)
        mask_moving = np.zeros_like(mov_arr)
        for slice_idx in range(fix_arr.shape[2]):
            mask_fixed[:, :, slice_idx] = extract_lung_mask(fix_arr[:, :, slice_idx], threshold_value=0.05)
            mask_moving[:, :, slice_idx] = extract_lung_mask(mov_arr[:, :, slice_idx], threshold_value=0.005)


        #reshape to model input
        # fix_arr = resize(fix_arr, (self.img_size[0], self.img_size[1], fix_arr.shape[2]), anti_aliasing=True)
        # mov_arr = resize(mov_arr, (self.img_size[0], self.img_size[1], mov_arr.shape[2]), anti_aliasing=True)

        """save copy of mask for vis"""
        # movImg_1dim = nib.Nifti1Image(mask_moving, aff_mov)
        # fixImg_1dim = nib.Nifti1Image(mask_fixed, aff_mov)
        # os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
        # nib.save(movImg_1dim, os.path.join(output_dir, 'vis', 'mov_mask.nii.gz'))
        # nib.save(fixImg_1dim, os.path.join(output_dir, 'vis', 'fix_mask.nii.gz'))

        # movImg_1dim = nib.Nifti1Image(mov_arr, aff_mov)
        # fixImg_1dim = nib.Nifti1Image(fix_arr, aff_mov)
        # os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
        # nib.save(movImg_1dim, os.path.join(output_dir, 'vis', 'mov_proc.nii.gz'))
        # nib.save(fixImg_1dim, os.path.join(output_dir, 'vis', 'fix_proc.nii.gz'))
        # sys.exit()


        return mov_arr, fix_arr, slices_to_keep_indices, orig_chunked_shape , mask_fixed, mask_moving

    def load_model(self, dir):
        """
        Loads a pre-trained DINOv2 model from a local file path.
        This version assumes the model has been downloaded manually.
        """
        # Define the expected path for the pre-trained model file
        model_fn = os.path.join(dir, "dinov2_vitl14_reg4_pretrain.pth")
    
        # --- MODIFICATION: Replaced download logic with an error check ---
    
        # Check if the model file exists at the expected path
        if not os.path.exists(model_fn):
            # If the file is not found, raise an error with instructions
            error_message = (
                f"Model file not found at '{model_fn}'.\n"
                f"Please download the model manually from:\n"
                f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth\n"
                f"And place it in the '{os.path.dirname(model_fn)}' directory."
            )
            raise FileNotFoundError(error_message)
        else:
            print(f"DINOv2 model found at '{model_fn}'.")

        # The rest of the function remains the same
        conf_fn = '{0:s}/dinov2/configs/eval/vitl14_reg4_pretrain'.format(sys.path[0])
        self.patch_size = 14
        self.embed_dim = 1024
    
        conf = load_and_merge_config(conf_fn)
        model = build_model_for_eval(conf, model_fn)
        model.to(device=torch.device('cuda', self.device_id))
    
        return model

    def encode_3D_gap(self, input_arr, gap=3):


        imageH, imageW, slice_num = input_arr.shape

        """old resize"""
        # feature_height = int(imageH * upsample_factor + 0.5)  // self.patch_size
        # feature_width = int(imageW * upsample_factor + 0.5) // self.patch_size
        # self.slice_num = slice_num
        # self.feature_height = feature_height
        # self.feature_width = feature_width

        """new uniform resize"""
        feature_height = self.feature_height
        feature_width = self.feature_width
        self.slice_num = slice_num


        input_arr = resize(input_arr, (feature_height*self.patch_size, feature_width*self.patch_size, slice_num), anti_aliasing=True)

        print(self.patch_size)
        print(feature_height, feature_width, slice_num)
        print('resized input shape', input_arr.shape)

        # 3D image into 2D model, stack each slices feature
        img_feature = np.zeros([feature_height * feature_width, slice_num, self.embed_dim])
        encoding_slice_idx = np.arange(0, slice_num-1, gap).tolist()
        encoding_slice_idx.append(slice_num-1)

        prev_slice = 0
        for slice_id in encoding_slice_idx:
            input_slice = input_arr[:, :, slice_id, np.newaxis]
            input_slice = np.repeat(input_slice, 3, axis=2)
            featrure = self.extract_dinov2_feature(input_slice)
            featrure = einops.rearrange(featrure, '1 n c -> n c')
            print("\rslice id:{} feature shape:{} ".format(slice_id, featrure.shape), end="")
            img_feature[:, slice_id, :] = featrure

            #interpolating the feature of the skipped slices
            if slice_id > 0 and slice_id < slice_num-1:
                for i in range(1, gap):
                    slice_id_gap = slice_id - i
                    if slice_id_gap >= 0:
                        featrure_gap = (featrure * (gap - i) + img_feature[:, prev_slice, :] * i) / gap
                        img_feature[:, slice_id_gap, :] = featrure_gap
            elif slice_id == slice_num-1:
                last_gap = slice_num - encoding_slice_idx[-2]
                for i in range(1, last_gap):
                    slice_id_gap = slice_num - i
                    featrure_gap = (featrure * (last_gap - i) + img_feature[:, prev_slice, :] * i) / last_gap
                    img_feature[:, slice_id_gap, :] = featrure_gap
            prev_slice = slice_id

        img_feature = img_feature.reshape([feature_height * feature_width * slice_num, self.embed_dim])


        return img_feature


    def extract_slice_feature(self, input_arr_orig, mask=True):

        """input single slice 2d, output the feature of that slice"""

        input_arr = resize(input_arr_orig, (self.feature_height*self.patch_size, self.feature_width*self.patch_size), anti_aliasing=True)
        if mask:
            input_arr_masksize = resize(input_arr_orig, (self.feature_height, self.feature_width), anti_aliasing=True)
            pca_mask = extract_lung_mask(input_arr_masksize).flatten().astype(bool)

        input_slice = input_arr[:, :, np.newaxis]
        input_slice = np.repeat(input_slice, 3, axis=2)
        featrure = self.extract_dinov2_feature(input_slice)

        featrure = einops.rearrange(featrure, '1 n c -> n c')

        if mask:
            return featrure, pca_mask
        return featrure, np.ones(featrure.shape[0], dtype=bool)
    

def jacobian_determinant(disp):
    # This is your provided function
    _, _, H, W, D = disp.shape
    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)
    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)
    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)
    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (
                jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) - \
             jacobian[1, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :,
                                                                                                       :, :]) + \
             jacobian[2, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :,
                                                                                                       :, :])
    return jacdet

def compute_dice_coefficient(mask1, mask2):
    """
    Compute the Dice Similarity Coefficient between two binary masks.
    """
    intersection = np.sum(mask1 * mask2)
    size1 = np.sum(mask1)
    size2 = np.sum(mask2)
    return (2.0 * intersection + 1e-5) / (size1 + size2 + 1e-5)


def compute_label_wise_dice(seg1, seg2, labels):
    """Compute Dice for each label/organ."""
    dice_results = []
    for label in labels:
        # Isolate current label in both segmentations
        seg1_label = seg1 == label
        seg2_label = seg2 == label
        
        # Compute Dice for the current label
        dice = compute_dice_coefficient(seg1_label, seg2_label)
        dice_results.append(dice)
    
    return dice_results


def compute_95_hausdorff_distance(seg1, seg2):
    # Assuming seg1 and seg2 are binary segmentation masks
    u_indices = np.array(np.where(seg1)).T
    v_indices = np.array(np.where(seg2)).T

    # Compute all pairwise distances between the two sets of points
    distances = cdist(u_indices, v_indices, 'euclidean')

    # Flatten the distance matrix and sort the distances
    sorted_distances = np.sort(distances, axis=None)

    # Find the 95th percentile distance
    hd_95 = np.percentile(sorted_distances, 95)
    return hd_95


def compute_label_wise_95hd(seg1, seg2, labels):
    hd95_results = []
    for label in labels:
        # Isolate current label in both segmentations
        seg1_label = seg1 == label
        seg2_label = seg2 == label

        # Compute 95% HD for the current label
        hd95 = compute_95_hausdorff_distance(seg1_label, seg2_label)
        hd95_results.append(hd95)

    return hd95_results

def score_case(seg_fixed, seg_moving, disp_field, aff_mov, case, label_list, output_dir, spacing=1):
    """
    Calculates per-organ Dice, HD95, sdLogJ, and the percentage of non-positive Jacobians.
    Following the same pattern as baseline datasets for consistent evaluation.
    """
    # The jacobian_determinant function returns the determinant for each voxel
    jac_det = jacobian_determinant(disp_field[np.newaxis, :, :, :, :])

    # Calculate sdLogJ (as in the original function)
    log_jac_det = np.log(jac_det.clip(1e-9, 1e9))
    sd_log_jac = log_jac_det[2:-2, 2:-2, 2:-2].std()

    # Calculate the percentage of non-positive Jacobians (folding)
    non_pos_jac_percent = np.sum(jac_det <= 0) / jac_det.size * 100

    # Apply deformation to moving segmentation for evaluation
    warped_seg_moving = map_coordinates(seg_moving, 
                                      np.meshgrid(np.arange(seg_moving.shape[0]), 
                                                np.arange(seg_moving.shape[1]), 
                                                np.arange(seg_moving.shape[2]), 
                                                indexing='ij') + disp_field, 
                                      order=0)  # Use nearest neighbor for labels

    # Per-organ Dice computation (following baseline dataset pattern)
    dice_coefficient = compute_label_wise_dice(seg_fixed, warped_seg_moving, label_list)
    dice_coefficient = np.array(dice_coefficient)

    # Per-organ HD95 computation  
    hd95 = compute_label_wise_95hd(seg_fixed, warped_seg_moving, label_list)
    hd95 = np.array(hd95)

    return {
        'DICE': dice_coefficient,
        'LogJacDetStd': sd_log_jac,
        'HD95': hd95,
        'NonPosJac': non_pos_jac_percent
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Dataset name")
    parser.add_argument("--csv_dir", type=str, default=None, help="Path to csv folder")
    parser.add_argument("--test_dir", type=str, default=None, help="Path to test folder")
    args = parser.parse_args()

    time_start = time.time()
    save_feature = False
    configs = {
        'smooth_weight': 2, 'lr': 3, 'num_iter': 1000, 'fm_downsample': 1,
        'feature_size': (80, 70), 'useSavedPCA': False, 'DINOReg_useMask': True,
        'window': True, 'convex': False, 'ztrans': False, 'iter_smooth_num': 5,
        'iter_smooth_kernel': 7, 'final_upsample': 1, 'mask': 'slice fill stack'
    }
    output_dir_0 = f'output/dinoreg-{configs["smooth_weight"]}smooth-{configs["num_iter"]}iter-itersmoothK{configs["iter_smooth_kernel"]}R{configs["iter_smooth_num"]}-lr3-fmd1-fmsize112x96-noconvex'
    print('output_dir_0', output_dir_0)
    dataset_dir = args.test_dir

    if configs['useSavedPCA']:
        PCA_matrix = np.load('/sample_dir/pca_matrix_AMOS_150x129_mask.npy')

    os.makedirs(output_dir_0, exist_ok=True)
    import json
    with open(os.path.join(output_dir_0, 'configs.json'), 'w') as f:
        json.dump(configs, f)

    pair_list = []
    with open(path.join(args.csv_dir, f"{args.data}_pairs.csv"), 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            pair_list.append(row)

    quantify = True
    dinoReg = dinoReg(args.csv_dir, lr=configs['lr'], smooth_weight=configs['smooth_weight'], num_iter=configs['iter_smooth_num'], feat_size=configs['feature_size'])

    exp_note = f'{args.data}_eval'
    output_dir = os.path.join(output_dir_0, exp_note)
    os.makedirs(output_dir, exist_ok=True)

    # NEW: Initialize lists for all metrics
    DICE_list = []
    LogJacDetStd_list = []
    hd_list = []
    NonPosJac_list = []
    InferenceTime_list = []
    eigenvalue_array = []

    csv_file = os.path.join(args.csv_dir, f"{args.data}_structures.csv")
    df = pd.read_csv(csv_file, header=None)
    label_list = df.iloc[0, :].tolist()

    for i, pair in enumerate(pair_list):
        print('case', i)
        patient_id = pair[0]
        patient_path = path.join(dataset_dir, patient_id)
        try:
            ct_file = [f for f in os.listdir(path.join(patient_path, 'CT')) if f.endswith('.nii.gz')][0]
            mr_file = [f for f in os.listdir(path.join(patient_path, 'MR')) if f.endswith('.nii.gz')][0]
            ct_seg_file = [f for f in os.listdir(path.join(patient_path, 'CT_seg')) if f.endswith('.nii.gz')][0]
            mr_seg_file = [f for f in os.listdir(path.join(patient_path, 'MR_seg')) if f.endswith('.nii.gz')][0]
            moving_img_filepath = path.join(patient_path, 'MR', mr_file)
            fixed_img_filepath = path.join(patient_path, 'CT', ct_file)
            moving_seg_filepath = path.join(patient_path, 'MR_seg', mr_seg_file)
            fixed_seg_filepath = path.join(patient_path, 'CT_seg', ct_seg_file)
        except (FileNotFoundError, IndexError) as e:
            print(f"  --> Skipping patient {patient_id}. Reason: {e}")
            continue

        moving_basename = f"{patient_id}_CT"
        fixed_basename = f"{patient_id}_MR"

        img_fixed = nib.load(fixed_img_filepath)
        img_moving = nib.load(moving_img_filepath)
        arr_fixed_orig, arr_moving_orig = img_fixed.get_fdata(), img_moving.get_fdata()
        aff_mov = img_moving.affine
        
        # --- NEW: Downsample the image arrays ---
        # Calculate the zoom factor to reach a target size (e.g., 128 in the largest dim)
        target_shape = np.array([128, 128, 128])
        zoom_factor = np.min(target_shape / np.array(arr_moving_orig.shape))

        print(f"  Downsampling images with factor: {zoom_factor:.2f}")
        arr_moving = scipy.ndimage.zoom(arr_moving_orig, zoom_factor, order=1)
        arr_fixed = scipy.ndimage.zoom(arr_fixed_orig, zoom_factor, order=1)
        # --- End of new code ---


        if os.path.exists(fixed_seg_filepath) and os.path.exists(moving_seg_filepath):
            seg_fixed = nib.load(fixed_seg_filepath).get_fdata()
            seg_moving = nib.load(moving_seg_filepath).get_fdata()
            # Use order=0 (nearest neighbor) for label maps to preserve integer labels
            seg_moving = scipy.ndimage.zoom(seg_moving, zoom_factor, order=0)
            seg_fixed = scipy.ndimage.zoom(seg_fixed, zoom_factor, order=0)
        else:
            seg_fixed, seg_moving = None, None

        # NEW: Measure inference time per case
        inference_start_time = time.time()
        
        # This is the core registration call
        H,W,D = arr_moving.shape
        identity = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D)).permute(0,4,1,2,3)
        disp_init = identity.numpy()

        disp_init = None
        load_note = None
        # load_note = 'cls_select'
        if load_note is not None:
            disp_init = nib.load(os.path.join(output_dir_0, load_note, i + '_disp_{}.nii.gz'.format(load_note))).get_fdata()
            disp_init = np.moveaxis(disp_init, 3, 0)[np.newaxis, :, :, :, :]
        
        # ---- DEBUG ----
        if i == 0: # Only do this for the first case
            print("  DEBUG: Saving preprocessed files for case 0...")
            debug_dir = os.path.join(output_dir, 'debug')
            os.makedirs(debug_dir, exist_ok=True)
    
            # Save the downsampled images and segmentations
            nib.save(nib.Nifti1Image(arr_fixed, aff_mov), os.path.join(debug_dir, 'debug_case0_fixed_img.nii.gz'))
            nib.save(nib.Nifti1Image(arr_moving, aff_mov), os.path.join(debug_dir, 'debug_case0_moving_img.nii.gz'))
    
            if seg_fixed is not None:
                nib.save(nib.Nifti1Image(seg_fixed, aff_mov), os.path.join(debug_dir, 'debug_case0_fixed_seg.nii.gz'))
                nib.save(nib.Nifti1Image(seg_moving, aff_mov), os.path.join(debug_dir, 'debug_case0_moving_seg.nii.gz'))
        # ---- DEBUG ----
    
        disp = dinoReg.case_inference(arr_moving, arr_fixed, arr_moving.shape, aff_mov, case_id=fixed_basename, 
                                      disp_init=disp_init, grid_sp_adam=configs['fm_downsample'], DINOReg_useMask=configs['DINOReg_useMask'])
        
        InferenceTime_list.append(time.time() - inference_start_time)
        print(f"  Inference Time: {InferenceTime_list[-1]:.2f} seconds")

        disp_img = nib.Nifti1Image(disp, aff_mov)
        nib.save(disp_img, os.path.join(output_dir, f'{moving_basename}_to_{fixed_basename}_disp_{exp_note}.nii.gz'))
        
        disp_for_scoring = np.moveaxis(disp, 3, 0)

        warped_image = map_coordinates(arr_moving, np.meshgrid(np.arange(arr_moving.shape[0]), np.arange(arr_moving.shape[1]), np.arange(arr_moving.shape[2]), indexing='ij') + disp_for_scoring, order=0)
        nib.save(nib.Nifti1Image(warped_image, aff_mov), os.path.join(output_dir, f'{moving_basename}_to_{fixed_basename}_warped_{exp_note}.nii.gz'))

        # --- Save overlays and visualizations ---
        # Save overlay: fixed CT + label
        save_overlay_png(arr_fixed, seg_fixed, os.path.join(output_dir, f'{patient_id}_fixed_CT_overlay.png'))
        # Save overlay: moving MR + label
        save_overlay_png(arr_moving, seg_moving, os.path.join(output_dir, f'{patient_id}_moving_MR_overlay.png'))
        # Save overlay: warped MR + warped label
        warped_seg_moving = map_coordinates(seg_moving, np.meshgrid(np.arange(seg_moving.shape[0]), np.arange(seg_moving.shape[1]), np.arange(seg_moving.shape[2]), indexing='ij') + disp_for_scoring, order=0)
        save_overlay_png(warped_image, warped_seg_moving, os.path.join(output_dir, f'{patient_id}_warped_MR_overlay.png'))
        # Save deformation field visualization
        save_deformation_field_png(disp_for_scoring, os.path.join(output_dir, f'{patient_id}_deformation_field_middle.png'))
        # Save error map
        save_error_map_png(warped_seg_moving, seg_fixed, os.path.join(output_dir, f'{patient_id}_error_map_middle.png'))
        
        if quantify and seg_fixed is not None:
            # NEW: Call the updated scoring function
            result = score_case(seg_fixed, seg_moving, disp_for_scoring, aff_mov, i, label_list, output_dir)

            # NEW: Append all the calculated metrics
            DICE_list.append(result['DICE'])
            LogJacDetStd_list.append(result['LogJacDetStd'])
            hd_list.append(result['HD95'])
            NonPosJac_list.append(result['NonPosJac']) # New
            
            print(f"  Overall DICE: {np.mean(result['DICE']):.4f}")
            print(f"  LogJacDetStd: {result['LogJacDetStd']:.4f}")
            print(f"  HD95: {np.mean(result['HD95']):.4f}")
            print(f"  Non-Positive Jacobian (%): {result['NonPosJac']:.4f}")
            
            # Print per-organ metrics for this case if 4 or 1 label
            if len(label_list) in [1, 4]:
                organ_metrics_str = ", ".join([f"Organ {label}: {result['DICE'][idx]:.4f}" 
                                             for idx, label in enumerate(label_list)])
                print(f"    Per-organ Dice -> {organ_metrics_str}")

    if quantify and DICE_list:
        # Calculate and print summary stats following baseline dataset pattern
        dice_array = np.array(DICE_list)
        hd_array = np.array(hd_list)
        num_param_array = np.array([0 for _ in DICE_list])  # Placeholder: set to 0, update if available
        
        # Overall metrics
        mean_dice = np.mean(DICE_list)
        std_dice = np.std(DICE_list)
        
        # Per-organ metrics (following baseline dataset pattern)
        dice_mean_by_organ = np.nanmean(dice_array, axis=0)
        dice_std_by_organ = np.nanstd(dice_array, axis=0)
        
        # Other metrics
        mean_logjacdet = np.mean(np.asarray(LogJacDetStd_list))
        std_logjacdet = np.std(np.asarray(LogJacDetStd_list))
        mean_nonposjac = np.mean(np.asarray(NonPosJac_list))
        std_nonposjac = np.std(np.asarray(NonPosJac_list))
        mean_hd95 = np.nanmean(hd_array)
        std_hd95 = np.nanstd(hd_array)
        mean_inf_time = np.mean(np.asarray(InferenceTime_list))
        std_inf_time = np.std(np.asarray(InferenceTime_list))
        mean_num_param = np.mean(num_param_array)
        std_num_param = np.std(num_param_array)

        print("\n--- Final Results ---")
        print(f"Overall DICE: {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"DICE mean by organ: {dice_mean_by_organ}")
        print(f"DICE std by organ: {dice_std_by_organ}")
        print(f"HD95: {mean_hd95:.4f} ± {std_hd95:.4f}")
        print(f"sdLogJ: {mean_logjacdet:.4f} ± {std_logjacdet:.4f}")
        print(f"Non-Positive Jacobian (%): {mean_nonposjac:.4f} ± {std_nonposjac:.4f}")
        print(f"Inference Time (s): {mean_inf_time:.2f} ± {std_inf_time:.2f}")
        print(f"Num Parameters: {mean_num_param:.2f} ± {std_num_param:.2f}")
        
        print(f"\nPer-Organ Dice Results:")
        if len(label_list) in [1, 4]:
            for idx, label in enumerate(label_list):
                print(f"  Organ {label}: {dice_mean_by_organ[idx]:.4f} ± {dice_std_by_organ[idx]:.4f}")
        else:
            print("  (Per-organ Dice not shown for this dataset)")

        # Save results to a text file (with per-organ details)
        summary_path = os.path.join(output_dir, f'summary_{exp_note}.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Metric, Mean, Std\n")
            f.write(f"Overall_DICE, {mean_dice:.4f}, {std_dice:.4f}\n")
            # Write per-organ results only for 1 or 4 label datasets
            if len(label_list) in [1, 4]:
                for idx, label in enumerate(label_list):
                    f.write(f"Organ_{label}_DICE, {dice_mean_by_organ[idx]:.4f}, {dice_std_by_organ[idx]:.4f}\n")
            f.write(f"HD95, {mean_hd95:.4f}, {std_hd95:.4f}\n")
            f.write(f"sdLogJ, {mean_logjacdet:.4f}, {std_logjacdet:.4f}\n")
            f.write(f"NonPosJac (%), {mean_nonposjac:.4f}, {std_nonposjac:.4f}\n")
            f.write(f"InferenceTime (s), {mean_inf_time:.2f}, {std_inf_time:.2f}\n")
            f.write(f"NumParameters, {mean_num_param:.2f}, {std_num_param:.2f}\n")

    print(f'\nTotal script time: {time.time() - time_start:.2f} seconds')
    print(f'Results saved in: {output_dir}')