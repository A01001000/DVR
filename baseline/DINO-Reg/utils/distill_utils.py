
import numpy as np
import torch
from dinov2.eval.setup import build_model_for_eval
from dinov2.configs import load_and_merge_config
import torchvision.transforms as tt
import nibabel as nib
from skimage.transform import resize
import sys
import einops
from utils.img_operations import remove_uniform_intensity_slices, pca_lowrank_transform, extract_lung_mask
import torch.nn as nn
import torch.nn.functional as F

class dinoReg:

    def __init__(self, device_id, lr=1, smooth_weight=10, num_iter=1000, feat_size=(80,80)):
        self.device_id = device_id
        self.patch_size = 16
        self.transform = tt.Compose([tt.Normalize(mean=0.5, std=0.2)])
        self.patch_grid_size = 32
        self.patch_margin = 10
        self.src_slice_num = 2
        self.patch_grid_h, self.patch_grid_w = 8, 8
        self.slice_step = 5
        # self.img_size = (int(256 / self.patch_size+0.5) * self.patch_size * 3, int(192 / self.patch_size+0.5) * self.patch_size * 3)  #
        # self.img_size = (2080, 2560)  #set size
        # self.img_size = (1120, 1120)  #set size
        self.embed_dim = 384
        self.model = self.load_model()
        self.img_size = (self.patch_size*feat_size[0], self.patch_size*feat_size[1])  #set size
        self.num_iter = num_iter


        self.batch_size = 12 #todo: implement parallel?
        # self.reg_featureDim = 1
        self.reg_featureDim = 24
        self.lr = lr
        self.smooth_weight = smooth_weight

        self.feature_height = feat_size[0]
        self.feature_width = feat_size[1]

    def extract_dinov2_feature(self, input_array):

        assert len(input_array.shape) == 3  # 2D image

        """flipping the input if needed"""
        # input_array = np.swapaxes(input_array, 0,1)

        input_rgb_array = input_array[np.newaxis, :, :, :]

        input_tensor = torch.Tensor(np.transpose(input_rgb_array, [0, 3, 1, 2]))
        input_tensor = self.transform(input_tensor)
        feature_array = self.model.forward_features(input_tensor.to(device=torch.device(self.device_id)))[
            'x_norm_patchtokens'].detach().cpu().numpy()
        del input_tensor

        return feature_array

    def case_inference(self, mov_arr, fix_arr, orig_img_shape, aff_mov,
                       mask_fixed=None, mask_moving=None, case_id='noID', disp_init=None, grid_sp_adam=1):

        assert len(mov_arr.shape) == 3

        """prepcocessing and feature extraction"""
        mov_arr, fix_arr, slices_to_keep_indices, orig_chunked_shape, mask_fixed_arr, mask_moving_arr = self.case_preprocess(mov_arr, fix_arr, mask_fixed, mask_moving)


        print('preprocessed moving and fixed image, shape', mov_arr.shape, fix_arr.shape)
        gap = 3 #3
        view = 'axial' #coronal by default, sagittal or axial

        mov_feature = self.encode_3D_gap_view(mov_arr, gap=gap, view=view)
        # mov_feature = self.encode_3D_gap_view_transpose(mov_arr, gap=gap, view=view)
        # mov_feature = self.encode_cat_gap_view(mov_arr, gap=gap, view=view)
        print('encoded moving image')
        fix_feature = self.encode_3D_gap_view(fix_arr, gap=gap, view=view)
        # fix_feature = self.encode_3D_gap_view_transpose(fix_arr, gap=gap, view=view)
        # fix_feature = self.encode_cat_gap_view(fix_arr, gap=gap, view=view)
        print('encoded fixed image')

        feat_sliceNum = self.slice_num

        DINOReg_useMask = False
        """PCA reduce dimension"""
        #only features inside the mask
        if DINOReg_useMask:
            # reshape to model output

            if view == 'axial':
                mask_fixed_arr = einops.rearrange(mask_fixed_arr, 'h w s -> h s w')
                mask_moving_arr = einops.rearrange(mask_moving_arr, 'h w s -> h s w')
            elif view == 'sagittal':
                mask_fixed_arr = einops.rearrange(mask_fixed_arr, 'h w s -> s w h')
                mask_moving_arr = einops.rearrange(mask_moving_arr, 'h w s -> s w h')
            elif view == 'coronal':
                pass
            else:
                raise ValueError('view should be axial, sagittal or coronal')

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
        # object_pca = PCA(n_components=self.reg_featureDim) #what is SVD solver?
        # reduced_patches = object_pca.fit_transform(all_features)

        reduced_patches, eigenvalues = pca_lowrank_transform(all_features, self.reg_featureDim)

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



        if view == 'axial':
            mov_pca = einops.rearrange(mov_pca, 'h s w c -> h w s c')
            fix_pca = einops.rearrange(fix_pca, 'h s w c -> h w s c')

        elif view == 'sagittal':
            mov_pca = einops.rearrange(mov_pca, 's w h c -> h w s c')
            fix_pca = einops.rearrange(fix_pca, 's w h c -> h w s c')
        elif view == 'coronal':
            pass
        else:
            raise ValueError('view should be axial, sagittal or coronal')

        """reconstruct features into original image shape"""
        mov_pca = einops.rearrange(mov_pca, 'h w s c -> h s w c')
        fix_pca = einops.rearrange(fix_pca, 'h w s c -> h s w c')


        print('reshaping to original image shape')
        mov_pca_rescaled = resize(mov_pca, (orig_chunked_shape[0], orig_chunked_shape[2], orig_chunked_shape[1], self.reg_featureDim),
                                   anti_aliasing=True)
        fix_pca_rescaled = resize(fix_pca, (orig_chunked_shape[0], orig_chunked_shape[2], orig_chunked_shape[1], self.reg_featureDim),
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

        """ConvexAdam optimization"""
        print('starting ConvexAdam optimization')

        # disp = convex_adam_3d(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled,
        # disp = convex_adam_3d_interSmooth(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled, #default 1000 iter
        #                       loss_func = "SSD", selected_niter=self.num_iter, lr=self.lr, selected_smooth=20, ic=True, lambda_weight=self.smooth_weight, disp_init=disp_init)
                              # loss_func = "SSD", selected_niter=5000, lr=self.lr, selected_smooth=3, ic=True, lambda_weight=self.smooth_weight, disp_init=disp_init)
                              # loss_func = "SSD", selected_niter=1000, lr=1, selected_smooth=3, ic=True, lambda_weight=10, disp_init=disp_init)

        disp = convex_adam_3d_param(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled, loss_func = "SSD", grid_sp_adam=grid_sp_adam,
                                               lambda_weight=self.smooth_weight, selected_niter=self.num_iter, lr=self.lr, disp_init=disp_init,
                                                iter_smooth_kernel = 7,
                                                iter_smooth_num = 3, end_smooth_kernel=1)
        
        # disp = convex_adam_3d_param_dataSmooth(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled, selected_smooth=0, loss_func = "SSD", grid_sp_adam=grid_sp_adam,
        #                                        lambda_weight=self.smooth_weight, selected_niter=self.num_iter, lr=self.lr, disp_init=disp_init,
        #                                         iter_smooth_kernel = 5,
        #                                         iter_smooth_num = 3)
        


        # disp = convex_adam_3d_interSmooth(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled,
        #                       loss_func = "SSD", selected_niter=1000, lr=1, selected_smooth=3, ic=True, lambda_weight=10, disp_init=disp_init)

        # disp = convex_adam_3d(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled, loss_func = "SSD", selected_niter=80, lr=3, selected_smooth=3, ic=True, lambda_weight=1.25)
        # disp = convex_adam_3d(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled, loss_func = "GPTNCC", selected_niter=80, lr=3, ic=False)
        # disp = convex_adam_3d_w0(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled, loss_func = "NCC_w0", selected_niter=200, lr=3, ic=False)
        # disp = convex_adam_3d_w0(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled,
        #                       loss_func = "SSD", selected_niter=800, lr=1, selected_smooth=3, ic=True, lambda_weight=10, disp_init=disp_init)

        """apply displacement field to moving image or landmarks"""

        """save copy of 1 channel feature for vis"""
        # mov_feat_1dim = mov_fullImg_pca_rescaled[:,:,:,:3]
        # fix_feat_1dim = fix_fullImg_pca_rescaled[:,:,:,:3]
        # movImg_1dim = nib.Nifti1Image(mov_feat_1dim, aff_mov)
        # fixImg_1dim = nib.Nifti1Image(fix_feat_1dim, aff_mov)
        # os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
        # nib.save(movImg_1dim, os.path.join(output_dir, 'vis', case_list[i] + '_mov_feat_gap2.nii.gz'))
        # nib.save(fixImg_1dim, os.path.join(output_dir, 'vis',case_list[i] + '_fix_feat_gap2.nii.gz'))
        # sys.exit()

        return disp

    def case_preprocess(self, mov_arr, fix_arr, mask_fixed, mask_moving):
        assert len(mov_arr.shape) == 3
        assert len(fix_arr.shape) == 3

        pad_indices = []
        filtered_image_data, slices_to_keep_indices = remove_uniform_intensity_slices(fix_arr)
        pad_indices.append(slices_to_keep_indices)
        fix_arr = filtered_image_data
        mov_arr = mov_arr[:, :, slices_to_keep_indices]

        if mask_fixed is not None:
            mask_fixed = mask_fixed[:, :, slices_to_keep_indices]
            mask_moving = mask_moving[:, :, slices_to_keep_indices]

        print('slices to keep', slices_to_keep_indices)

        fix_arr = einops.rearrange(fix_arr, 'h w s -> h s w')
        mov_arr = einops.rearrange(mov_arr, 'h w s -> h s w')

        #loaded mask
        # mask_fixed = einops.rearrange(mask_fixed, 'h w s -> h s w')
        # mask_moving = einops.rearrange(mask_moving, 'h w s -> h s w')

        orig_chunked_shape = fix_arr.shape

        # normalize to lung CT window
        #OASIS should not be normalized, maybe creat dataset specific preproc in future
        # fix_arr = MR_normalize(fix_arr, quantile=100)
        # mov_arr = MR_normalize(mov_arr, quantile=100)

        # window_level = -600, window_width = 1500
        # mov_arr = to_lungCT_window(mov_arr, wl=-600, ww=1500)
        # fix_arr = to_lungCT_window(fix_arr, wl=-600, ww=1500)

        #intensity mask
        mask_fixed = np.where(fix_arr > 0.05, 1.0, 0)
        mask_moving = np.where(mov_arr > 0.05, 1.0, 0)

        filtered_z = fix_arr.shape[2]

        #reshape to model input
        # fix_arr = resize(fix_arr, (self.img_size[0], self.img_size[1], fix_arr.shape[2]), anti_aliasing=True)
        # mov_arr = resize(mov_arr, (self.img_size[0], self.img_size[1], mov_arr.shape[2]), anti_aliasing=True)

        # movImg_1dim = nib.Nifti1Image(mov_arr, aff_mov)
        # fixImg_1dim = nib.Nifti1Image(fix_arr, aff_mov)
        # fix_arr_threshold = np.where(fix_arr > 0.05, 1.0, 0)
        # fixImg_1dim_threshold = nib.Nifti1Image(mask_moving, aff_mov)
        # os.makedirs(os.path.join(output_dir, 'img'), exist_ok=True)
        # nib.save(movImg_1dim, os.path.join(output_dir, 'img',case_list[i] + '_mov_feat.nii.gz'))
        # nib.save(fixImg_1dim, os.path.join(output_dir, 'img',case_list[i] + '_fix_feat.nii.gz'))
        # nib.save(fixImg_1dim_threshold, os.path.join(output_dir, 'vis',case_list[i] + '_threshold.nii.gz'))
        # sys.exit()

        return mov_arr, fix_arr, slices_to_keep_indices, orig_chunked_shape , mask_fixed, mask_moving

    def load_model(self):
        """load model"""
        # vitl-14
        # conf_fn = '{0:s}/dinov2/configs/eval/vitl14_pretrain'.format(sys.path[0])
        # model_fn = '/fast/songx/models/dinov2/dinov2_vitl14_pretrain.pth'

        conf_fn = '{0:s}/dinov2/configs/eval/vitl14_reg4_pretrain'.format(sys.path[0])
        model_fn = '/fast/songx/models/dinov2/dinov2_vitl14_reg4_pretrain.pth'
        self.patch_size = 14
        self.embed_dim = 1024

        # vits-14
        # conf_fn = '{0:s}/dinov2/configs/eval/vits14_reg4_pretrain'.format(sys.path[0])
        # model_fn = '/fast/songx/models/dinov2/dinov2_vits14_reg4_pretrain.pth'
        # self.patch_size = 14
        # self.embed_dim = 384

        # conf_fn = '{0:s}/dinov2/configs/eval/vitl16_xs'.format(sys.path[0])
        # model_fn = '/fast/songx/train_tempFiles/dino_ft/vitl14-100epoch/eval/training_199999/teacher_checkpoint.pth'
        # self.patch_size = 16

        # conf_fn = '{0:s}/dinov2/configs/eval/vits16_xs'.format(sys.path[0])
        # # model_fn = '/fast/songx/train_tempFiles/dino_ft/vits16-100epoch/eval/training_89199/teacher_checkpoint.pth'
        # model_fn = '/fast/songx/train_tempFiles/dino_ft/vits16_mask_100epoch/eval/training_159999/teacher_checkpoint.pth'
        # self.patch_size = 16

        # conf_fn = '{0:s}/dinov2/configs/eval/vitl14_xs1'.format(sys.path[0])
        # model_fn = '/fast/songx/train_tempFiles/dino_ft/vitl14-51epoch_save/eval/training_74999/teacher_checkpoint.pth'
        # self.patch_size = 14

        # conf_fn = '{0:s}/dinov2/configs/eval/vitl14_xs1'.format(sys.path[0])
        # model_fn = '/fast/songx/train_tempFiles/dino_ft/vitl14-51epoch_save/eval/training_74999/teacher_checkpoint.pth'


        conf = load_and_merge_config(conf_fn)
        model = build_model_for_eval(conf, model_fn)
        model.to(device=torch.device(self.device_id))
        return model

    def encode_3D_gap_view(self, input_arr, gap=3, view='coronal'):


        if view == 'axial':
            input_arr = einops.rearrange(input_arr, 'h w s -> h s w')
        elif view == 'sagittal':
            input_arr = einops.rearrange(input_arr, 'h w s -> s w h')
        elif view == 'coronal':
            pass
        else:
            raise ValueError('view should be axial, sagittal or coronal')

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
            # pca_mask = np.where(input_arr_masksize > 0.05, 1.0, 0).flatten().astype(bool)
            # print('mask size', input_arr_masksize.shape, 'max', input_arr_masksize.max(), 'min', input_arr_masksize.min())
            pca_mask = extract_lung_mask(input_arr_masksize).flatten().astype(bool)

        input_slice = input_arr[:, :, np.newaxis]
        input_slice = np.repeat(input_slice, 3, axis=2)
        featrure = self.extract_dinov2_feature(input_slice)

        featrure = einops.rearrange(featrure, '1 n c -> n c')

        if mask:
            return featrure, pca_mask
        return featrure, np.ones(featrure.shape[0], dtype=bool)
    


# def channelCrossEntLoss(prediction, target):
#     # Ensure prediction is of shape CxHxW and target is HxW
#     B, C, H, W, D = prediction.size()
    
#     # Flatten prediction to CxL, where L = H * W
#     prediction_flat = prediction.view(C, -1)
#     # Transpose prediction to LxC (L: number of pixels, C: number of classes)
#     prediction_flat = prediction_flat.transpose(0, 1)
    
#     # do same for target
#     target_flat = target.view(C, -1)
#     target_flat = target_flat.transpose(0, 1)
    
#     # Apply log_softmax along the class dimension
#     prediction_softmax = F.softmax(prediction_flat, dim=1)
#     target_softmax = F.softmax(target_flat, dim=1)
    
#     # Compute cross entropy loss
#     loss = - torch.sum(target_softmax * torch.log(prediction_softmax + 1e-8)) / target_softmax.size(0)
    
#     return loss


def channelCrossEntLoss(prediction, target):
    # Ensure prediction and target are both of shape BxCxHxWxD
    B, C, H, W, D = prediction.size()

    # Flatten both prediction and target to BxCxL, where L = H * W * D
    prediction_flat = prediction.view(B, C, -1)
    target_flat = target.view(B, C, -1)

    # Apply softmax along the class dimension (C) for the prediction and target
    prediction_softmax = F.softmax(prediction_flat, dim=1)
    target_softmax = F.softmax(target_flat, dim=1)
    
    # Compute cross-entropy loss manually
    loss = - torch.sum(target_softmax * torch.log(prediction_softmax + 1e-8)) / target_softmax.size(0)
    
    return loss