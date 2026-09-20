import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage.interpolation import zoom as zoom
from scipy.ndimage import distance_transform_edt as edt
from convex_adam_utils import *
import time
import argparse
import nibabel as nib
import os
import sys
from data_utils import get_files_paths
import scipy
from scipy.ndimage import map_coordinates
import einops
import warnings
warnings.filterwarnings("ignore")


# extract MIND and/or semantic nnUNet features
def extract_features(img_fixed,
                    img_moving,
                    mind_r,
                    mind_d,
                    use_mask,
                    mask_fixed,
                    mask_moving):

    # MIND features
    if use_mask:
        H,W,D = img_fixed.shape[-3:]

        #replicate masking
        avg3 = nn.Sequential(nn.ReplicationPad3d(1),nn.AvgPool3d(3,stride=1))
        avg3.cuda()
        
        mask = (avg3(mask_fixed.view(1,1,H,W,D).cuda())>0.9).float()
        _,idx = edt((mask[0,0,::2,::2,::2]==0).squeeze().cpu().numpy(),return_indices=True)
        fixed_r = F.interpolate((img_fixed[::2,::2,::2].cuda().reshape(-1)[idx[0]*D//2*W//2+idx[1]*D//2+idx[2]]).unsqueeze(0).unsqueeze(0),scale_factor=2,mode='trilinear')
        fixed_r.view(-1)[mask.view(-1)!=0] = img_fixed.cuda().reshape(-1)[mask.view(-1)!=0]

        mask = (avg3(mask_moving.view(1,1,H,W,D).cuda())>0.9).float()
        _,idx = edt((mask[0,0,::2,::2,::2]==0).squeeze().cpu().numpy(),return_indices=True)
        moving_r = F.interpolate((img_moving[::2,::2,::2].cuda().reshape(-1)[idx[0]*D//2*W//2+idx[1]*D//2+idx[2]]).unsqueeze(0).unsqueeze(0),scale_factor=2,mode='trilinear')
        moving_r.view(-1)[mask.view(-1)!=0] = img_moving.cuda().reshape(-1)[mask.view(-1)!=0]

        features_fix = MINDSSC(fixed_r.cuda(),mind_r,mind_d).half()
        features_mov = MINDSSC(moving_r.cuda(),mind_r,mind_d).half()
    else:
        img_fixed = img_fixed.unsqueeze(0).unsqueeze(0)
        img_moving = img_moving.unsqueeze(0).unsqueeze(0)
        features_fix = MINDSSC(img_fixed.cuda(),mind_r,mind_d).half()
        features_mov = MINDSSC(img_moving.cuda(),mind_r,mind_d).half()
    
    return features_fix, features_mov

# coupled convex optimisation with adam instance optimisation
def convex_adam(path_img_fixed,
                path_img_moving,
                mind_r=1,
                mind_d=2,
                lambda_weight=1.25,
                grid_sp=6,
                disp_hw=4,
                selected_niter=80,
                selected_smooth=0,
                grid_sp_adam=2,
                ic=True,
                use_mask=False,
                path_fixed_mask=None,
                path_moving_mask=None,
                result_path='./'):

    img_fixed = torch.from_numpy(nib.load(path_img_fixed).get_fdata()).float()
    img_moving = torch.from_numpy(nib.load(path_img_moving).get_fdata()).float()
    
    if use_mask:
        mask_fixed = torch.from_numpy(nib.load(path_fixed_mask).get_fdata()).float()
        mask_moving = torch.from_numpy(nib.load(path_moving_mask).get_fdata()).float()
    else: 
        mask_fixed = None
        mask_moving = None
    
    H,W,D = img_fixed.shape
    print('img_fixed shape: ', img_fixed.shape)


    torch.cuda.synchronize()
    t0 = time.time()

    #compute features and downsample (using average pooling)
    with torch.no_grad():      
        
        features_fix, features_mov = extract_features(img_fixed=img_fixed,
                                                        img_moving=img_moving,
                                                        mind_r=mind_r,
                                                        mind_d=mind_d,
                                                        use_mask=use_mask,
                                                        mask_fixed=mask_fixed,
                                                        mask_moving=mask_moving)

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import minmax_scale
        def reduction_PCA(temp_feature):
            object_pca = PCA(n_components=3)
            reduced_patches = object_pca.fit_transform(temp_feature)
            reduced_patches = minmax_scale(reduced_patches)
            return reduced_patches
        # print('features_mov shape: ', features_mov.shape)
        # features_mov = features_mov.cpu().numpy().squeeze()
        # features_mov = einops.rearrange(features_mov, 'c h w d -> (h w d) c')
        # mov_feat_3dim = reduction_PCA(features_mov.reshape(-1,features_mov.shape[-1]))
        # mov_feat_3dim = mov_feat_3dim.reshape(256,192,192,3)
        # movImg_1dim = nib.Nifti1Image(mov_feat_3dim, nib.load(path_img_fixed).affine)
        # nib.save(movImg_1dim, os.path.join(output_dir, case_list[i] + '_mov_feat3dim_MIND.nii.gz'))
        # sys.exit()
        
        features_fix_smooth = F.avg_pool3d(features_fix,grid_sp,stride=grid_sp)
        features_mov_smooth = F.avg_pool3d(features_mov,grid_sp,stride=grid_sp)

        # print(features_fix_smooth.shape)
        # sys.exit()

        n_ch = features_fix_smooth.shape[1]
    print('features_fix_smooth shape: ', features_fix_smooth.shape)
    print('features_fix shape: ', features_fix.shape)

    # compute correlation volume with SSD
    ssd,ssd_argmin = correlate(features_fix_smooth,features_mov_smooth,disp_hw,grid_sp,(H,W,D), n_ch)

    # provide auxiliary mesh grid
    disp_mesh_t = F.affine_grid(disp_hw*torch.eye(3,4).cuda().half().unsqueeze(0),(1,1,disp_hw*2+1,disp_hw*2+1,disp_hw*2+1),align_corners=True).permute(0,4,1,2,3).reshape(3,-1,1)
    
    # perform coupled convex optimisation
    disp_soft = coupled_convex(ssd,ssd_argmin,disp_mesh_t,grid_sp,(H,W,D))
    
    # if "ic" flag is set: make inverse consistent
    if ic:
        scale = torch.tensor([H//grid_sp-1,W//grid_sp-1,D//grid_sp-1]).view(1,3,1,1,1).cuda().half()/2

        ssd_,ssd_argmin_ = correlate(features_mov_smooth,features_fix_smooth,disp_hw,grid_sp,(H,W,D), n_ch)

        disp_soft_ = coupled_convex(ssd_,ssd_argmin_,disp_mesh_t,grid_sp,(H,W,D))
        disp_ice,_ = inverse_consistency((disp_soft/scale).flip(1),(disp_soft_/scale).flip(1),iter=15)

        disp_hr = F.interpolate(disp_ice.flip(1)*scale*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)
    
    else:
        disp_hr=disp_soft


    # run Adam instance optimisation
    if lambda_weight > 0:
        with torch.no_grad():

            patch_features_fix = F.avg_pool3d(features_fix,grid_sp_adam,stride=grid_sp_adam)
            patch_features_mov = F.avg_pool3d(features_mov,grid_sp_adam,stride=grid_sp_adam)


        #create optimisable displacement grid
        disp_lr = F.interpolate(disp_hr,size=(H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),mode='trilinear',align_corners=False)

        net = nn.Sequential(nn.Conv3d(3,1,(H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data/grid_sp_adam
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1)

        grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),align_corners=False)

        #run Adam optimisation with diffusion regularisation and B-spline smoothing
        for iter in range(selected_niter):
            optimizer.zero_grad()

            disp_sample = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(net[0].weight,3,stride=1,padding=1),3,stride=1,padding=1),3,stride=1,padding=1).permute(0,2,3,4,1)
            reg_loss = lambda_weight*((disp_sample[0,:,1:,:]-disp_sample[0,:,:-1,:])**2).mean()+\
            lambda_weight*((disp_sample[0,1:,:,:]-disp_sample[0,:-1,:,:])**2).mean()+\
            lambda_weight*((disp_sample[0,:,:,1:]-disp_sample[0,:,:,:-1])**2).mean()

            scale = torch.tensor([(H//grid_sp_adam-1)/2,(W//grid_sp_adam-1)/2,(D//grid_sp_adam-1)/2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1,3).cuda().float()+((disp_sample.view(-1,3))/scale).flip(1).float()

            patch_mov_sampled = F.grid_sample(patch_features_mov.float(),grid_disp.view(1,H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam,3).cuda(),align_corners=False,mode='bilinear')

            sampled_cost = (patch_mov_sampled-patch_features_fix).pow(2).mean(1)*12
            loss = sampled_cost.mean()
            (loss+reg_loss).backward()
            optimizer.step()

        fitted_grid = disp_sample.detach().permute(0,4,1,2,3)
        disp_hr = F.interpolate(fitted_grid*grid_sp_adam,size=(H,W,D),mode='trilinear',align_corners=False)

        if selected_smooth == 5:
            kernel_smooth = 5
            padding_smooth = kernel_smooth//2
            disp_hr = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(disp_hr,kernel_smooth,padding=padding_smooth,stride=1),kernel_smooth,padding=padding_smooth,stride=1),kernel_smooth,padding=padding_smooth,stride=1)

        if selected_smooth == 3:
            kernel_smooth = 3
            padding_smooth = kernel_smooth//2
            disp_hr = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(disp_hr,kernel_smooth,padding=padding_smooth,stride=1),kernel_smooth,padding=padding_smooth,stride=1),kernel_smooth,padding=padding_smooth,stride=1)

    torch.cuda.synchronize()
    t1 = time.time()
    case_time = t1-t0
    print('case time: ', case_time)
    #
    x = disp_hr[0,0,:,:,:].cpu().half().data.numpy()
    y = disp_hr[0,1,:,:,:].cpu().half().data.numpy()
    z = disp_hr[0,2,:,:,:].cpu().half().data.numpy()
    displacements = np.stack((x,y,z),3).astype(float)

    # affine = nib.load(path_img_fixed).affine
    # disp_nii = nib.Nifti1Image(displacements, affine)
    # nib.save(disp_nii, os.path.join(result_path,'disp_MIND.nii.gz'))
    return displacements

def compute_tre(x, y, spacing):
    return np.linalg.norm((x - y) * spacing, axis=1)

def jacobian_determinant(disp):
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

def score_case_invert(lms_fixed, lms_moving, disp_field, fixed_arr, spacing=1):

    print('disp shape in score_case',disp_field.shape)
    print('fixed_arr shape in score_case', fixed_arr.shape)
    # disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])

    jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(0.000000001, 1000000000)
    log_jac_det = np.log(jac_det)

    lms_fixed_disp_x = map_coordinates(disp_field[0], lms_moving.transpose())
    lms_fixed_disp_y = map_coordinates(disp_field[1], lms_moving.transpose())
    lms_fixed_disp_z = map_coordinates(disp_field[2], lms_moving.transpose())
    lms_fixed_disp = np.array((lms_fixed_disp_x, lms_fixed_disp_y, lms_fixed_disp_z)).transpose()

    lms_moving_warped = lms_moving + lms_fixed_disp
    # lms_moving_warped = lms_moving - lms_fixed_disp

    tre = compute_tre(lms_moving_warped, lms_fixed, spacing)

    print(np.ma.MaskedArray(log_jac_det, 1 - fixed_arr[2:-2, 2:-2, 2:-2]).std())

    return {'TRE': tre.mean(),
            'LogJacDetStd': np.ma.MaskedArray(log_jac_det, 1 - fixed_arr[2:-2, 2:-2, 2:-2]).std()}
def score_case(lms_fixed, lms_moving, disp_field, fixed_arr, spacing=1):

    print('disp shape in score_case',disp_field.shape)
    # disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])

    jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(0.000000001, 1000000000)
    log_jac_det = np.log(jac_det)

    lms_fixed_disp_x = map_coordinates(disp_field[0], lms_fixed.transpose())
    lms_fixed_disp_y = map_coordinates(disp_field[1], lms_fixed.transpose())
    lms_fixed_disp_z = map_coordinates(disp_field[2], lms_fixed.transpose())
    lms_fixed_disp = np.array((lms_fixed_disp_x, lms_fixed_disp_y, lms_fixed_disp_z)).transpose()

    lms_fixed_warped = lms_fixed + lms_fixed_disp

    tre = compute_tre(lms_fixed_warped, lms_moving, spacing)

    return {'TRE': tre.mean(),
            'LogJacDetStd': np.ma.MaskedArray(log_jac_det, 1 - fixed_arr[2:-2, 2:-2, 2:-2]).std()}

def to_lungCT_window(image_data, wl=-600, ww=1500):
    img = np.clip(image_data, wl - ww / 2, wl + ww / 2)
    # normalzie
    img = (img - (wl - ww / 2)) / ww
    return img


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    # parser.add_argument("-f","--path_img_fixed", type=str, required=True)
    # parser.add_argument("-f","--path_img_fixed", type=str, default='/fast/songx/datasets/AbdomenMRCT/imagesTs/AbdomenMRCT_0009_0000.nii.gz')
    # parser.add_argument("-f","--path_img_fixed", type=str, default='images/MR_slice_0015_0000_97_stack10.nii.gz')
    # parser.add_argument("-f","--path_img_fixed", type=str, default='/fast/songx/tempFiles/lungReg/vis/img/ThoraxCBCT_0000_0001_fix.nii.gz')
    parser.add_argument("-f","--path_img_fixed", type=str, default='/fast/songx/datasets/ThoraxCBCT/imagesTr/ThoraxCBCT_0000_0000.nii.gz')

    # parser.add_argument("-m",'--path_img_moving', type=str, required=True)
    parser.add_argument("-m",'--path_img_moving', type=str, default='/fast/songx/datasets/ThoraxCBCT/imagesTr/ThoraxCBCT_0000_0001.nii.gz')
    # parser.add_argument("-m",'--path_img_moving', type=str, default='images/CT_slice_0015_0001_118_stack10.nii.gz')
    parser.add_argument('--mind_r', type=int, default=1)
    parser.add_argument('--mind_d', type=int, default=2)
    parser.add_argument('--lambda_weight', type=float, default=1.25)
    parser.add_argument('--grid_sp', type=int, default=6)
    parser.add_argument('--disp_hw', type=int, default=4)
    parser.add_argument('--selected_niter', type=int, default=800)
    parser.add_argument('--selected_smooth', type=int, default=0)
    parser.add_argument('--grid_sp_adam', type=int, default=2)
    parser.add_argument('--ic', choices=('True','False'), default='True')
    parser.add_argument('--use_mask', choices=('True','False'), default='True')
    parser.add_argument('--path_mask_fixed', type=str, default='/fast/songx/tempFiles/lungReg/handy/image_mask.nii.gz')
    parser.add_argument('--path_mask_moving', type=str, default='/fast/songx/tempFiles/lungReg/handy/image_mask.nii.gz')
    parser.add_argument('--result_path', type=str, default='/fast/songx/tempFiles/lungReg')

    args= parser.parse_args()

    if args.ic == 'True':
        ic=True
    else:
        ic=False

    if args.use_mask == 'True':
        use_mask=True
    else:
        use_mask=False


    data_dir = '/fast/songx/datasets/Release_06_12_23'
    output_dir = '/fast/songx/tempFiles/lungReg/newShape'

    TRE_list = []
    LogJacDetStd_list = []

    # exp_note = 'MIND_init'
    exp_note = 'MIND_featureVis'
    output_dir = os.path.join(output_dir, exp_note)
    os.makedirs(output_dir, exist_ok=True)

    img_fixed_fn_all, img_moving_fn_all, mask_fixed_fn_all, mask_moving_fn_all, case_list, aff_mov_all, kpts_fixed_all, kpts_moving_all = get_files_paths(
        data_dir, 'ThoraxCBCT', 'Val')
        # data_dir, 'ThoraxCBCT', 'Tr')

    for i in range(len(case_list)):
        print('case', case_list[i])

        fixed_fn = img_fixed_fn_all[i]
        moving_fn = img_moving_fn_all[i]
        kpts_fixed = kpts_fixed_all[i]
        kpts_moving = kpts_moving_all[i]
        aff_mov = aff_mov_all[i]
        fixed_mask_fn = mask_fixed_fn_all[i]
        moving_mask_fn = mask_moving_fn_all[i]

        disp = convex_adam(fixed_fn,
                moving_fn,
                args.mind_r,
                args.mind_d,
                args.lambda_weight,
                args.grid_sp,
                args.disp_hw,
                800,
                args.selected_smooth,
                args.grid_sp_adam,
                ic,
                False,#use mask
                fixed_mask_fn,
                moving_mask_fn,
                output_dir)

        disp_img = nib.Nifti1Image(disp, aff_mov)
        nib.save(disp_img, os.path.join(output_dir, case_list[i] + '_disp_{}.nii.gz'.format(exp_note)))

        #load moving image
        mov_img = nib.load(moving_fn)
        mov_arr = mov_img.get_fdata()
        fix_img = nib.load(fixed_fn)
        fix_arr = fix_img.get_fdata()

        disp = np.moveaxis(disp, 3, 0)

        # warp moving image
        D, H, W = mov_arr.shape
        identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        warped_image = map_coordinates(mov_arr, identity + disp, order=0)
        warped_image = nib.Nifti1Image(warped_image, aff_mov)
        nib.save(warped_image, os.path.join(output_dir, case_list[i] + '_warped_{}.nii.gz'.format(exp_note)))

        # normalize arr_fixed
        arr_fixed_norm = to_lungCT_window(fix_arr)

        result = score_case(kpts_fixed, kpts_moving, disp, arr_fixed_norm)
        # result = score_case_invert(kpts_fixed, kpts_moving, disp, arr_fixed_norm)

        TRE_list.append(result['TRE'])
        LogJacDetStd_list.append(result['LogJacDetStd'])
        print('TRE', result['TRE'], 'LogJacDetStd', result['LogJacDetStd'])

    # print mean and std
    print('TRE mean', np.mean(np.asarray(TRE_list)), 'TRE std', np.std(np.asarray(TRE_list)))
    print('LogJacDetStd mean', np.mean(np.asarray(LogJacDetStd_list)), 'LogJacDetStd std',
          np.std(np.asarray(LogJacDetStd_list)))

    print(TRE_list)
    print(LogJacDetStd_list)

    np.savetxt(os.path.join(output_dir, 'TRE_list_{}.txt'.format(exp_note)), np.asarray(TRE_list), fmt='%.3f')
    np.savetxt(os.path.join(output_dir, 'LogJacDetStd_list_{}.txt'.format(exp_note)), np.asarray(LogJacDetStd_list),
               fmt='%.3f')

