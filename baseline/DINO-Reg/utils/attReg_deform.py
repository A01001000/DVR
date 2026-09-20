
import os
os.environ['VXM_BACKEND'] = 'pytorch'
from utils.voxelmorph.torch import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class NLBlockND_cross_noRes(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND_cross_noRes, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_thisBranch, x_otherBranch):
        # x_thisBranch for g and theta
        # x_otherBranch for phi
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        # print(x_thisBranch.shape)

        batch_size = x_thisBranch.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x_thisBranch).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x_thisBranch.view(batch_size, self.in_channels, -1)
            phi_x = x_otherBranch.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, -1)
            # theta_x = theta_x.permute(0, 2, 1)
            phi_x = phi_x.permute(0, 2, 1)
            f = torch.matmul(phi_x, theta_x)
            # print('f shape',f.shape)

        # elif self.mode == "concatenate":
        else:  # default as concatenate
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_thisBranch.size()[2:])

        W_y = self.W_z(y)

        # no residual connection
        return W_y


class attReg_deform(nn.Module):

    def __init__(self, inshape, inChannel=1):
        super().__init__()
        self.relu = nn.LeakyReLU(inplace=True)

        """layers for path global"""
        self.path1_block1_conv = nn.Conv3d(
            in_channels=inChannel,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path1_block1_conv_1 = nn.Conv3d(in_channels=32,out_channels=32,kernel_size=3,stride=(1, 1, 1),padding=1,bias=False)
        self.path1_block1_bn = nn.InstanceNorm3d(32)
        self.path1_block1_bn_1 = nn.InstanceNorm3d(32)
        self.maxpool_downsample_pathGlobal11 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.path1_block2_conv = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path1_block2_conv_1 = nn.Conv3d(in_channels=32,out_channels=32,kernel_size=3,stride=(1, 1, 1),padding=1,bias=False)

        self.path1_block2_bn = nn.InstanceNorm3d(32)
        self.path1_block2_bn_1 = nn.InstanceNorm3d(32)

        self.maxpool_downsample_pathGlobal12 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.path1_block3_conv = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path1_block3_conv_1 = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=3,stride=(1, 1, 1),padding=1,bias=False)

        self.path1_block3_bn = nn.InstanceNorm3d(64)
        self.path1_block3_bn_1 = nn.InstanceNorm3d(64)
        self.maxpool_downsample_pathGlobal13 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.path1_block3_NLCross = NLBlockND_cross_noRes(64)


        self.path2_block1_conv = nn.Conv3d(
            in_channels=inChannel,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path2_block1_conv_1 = nn.Conv3d(in_channels=32,out_channels=32,kernel_size=3,stride=(1, 1, 1),padding=1,bias=False)
        self.path2_block1_bn = nn.InstanceNorm3d(32)
        self.path2_block1_bn_1 = nn.InstanceNorm3d(32)
        self.maxpool_downsample_pathGlobal21 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.path2_block2_conv = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path2_block2_conv_1 = nn.Conv3d(in_channels=32,out_channels=32,kernel_size=3,stride=(1, 1, 1),padding=1,bias=False)

        self.path2_block2_bn = nn.InstanceNorm3d(32)
        self.path2_block2_bn_1 = nn.InstanceNorm3d(32)
        self.maxpool_downsample_pathGlobal22 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.path2_block3_conv = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path2_block3_conv_1 = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=3,stride=(1, 1, 1),padding=1,bias=False)

        self.path2_block3_bn = nn.InstanceNorm3d(64)
        self.path2_block3_bn_1 = nn.InstanceNorm3d(64)
        self.maxpool_downsample_pathGlobal23 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.path2_block3_NLCross = NLBlockND_cross_noRes(64)

        """new combined"""
        self.conv3d_7 = nn.Conv3d(in_channels=128, out_channels=32, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.pathC_bn1_1 = nn.InstanceNorm3d(128)
        self.pathC_bn1_2 = nn.InstanceNorm3d(128)

        self.upConv = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=2, stride=2)
        self.upConv_bn = nn.InstanceNorm3d(32)

        self.conv3d_quat_downChannel = nn.Conv3d(in_channels=128, out_channels=32, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.conv3d_8 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.conv3d_8_1 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.conv3d_8_bn = nn.InstanceNorm3d(64)
        self.conv3d_8_1_bn = nn.InstanceNorm3d(32)

        self.upConv2 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=2, stride=2)
        self.upConv2_bn = nn.InstanceNorm3d(32)
        self.conv3d_half_downChannel = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.conv3d_9 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.conv3d_9_1 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.conv3d_9_bn = nn.InstanceNorm3d(64)
        self.conv3d_9_1_bn = nn.InstanceNorm3d(16)

        # init flow layer with small weights and bias
        self.flow = nn.Conv3d(in_channels=16, out_channels=3, kernel_size=3, padding=1)

        # configure transformer
        # self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
        self.transformer = layers.SpatialTransformer(inshape)
        self.transformerForSeg = layers.SpatialTransformer(inshape, mode='nearest')

        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, moving_img, fixed_img):

        # moving_img =  images[:,0:1,...] #moving
        # fixed_img  =  images[:,1:2,...] #fixed

        #make copies of the images
        x_path1 = torch.clone(moving_img)
        x_path2 = torch.clone(fixed_img)

        """path 1"""
        x_path1 = self.path1_block1_conv(x_path1)
        x_path1 = self.path1_block1_bn(x_path1)
        x_path1 = self.relu(x_path1)
        x_path1 = self.path1_block1_conv_1(x_path1)
        x_path1 = self.path1_block1_bn_1(x_path1)
        x_path1 = self.relu(x_path1)
        x_path1 = self.maxpool_downsample_pathGlobal11(x_path1)

        x_path1 = self.path1_block2_conv(x_path1)
        x_path1 = self.path1_block2_bn(x_path1)
        x_path1 = self.relu(x_path1)
        x_path1 = self.path1_block2_conv_1(x_path1)
        x_path1 = self.path1_block2_bn_1(x_path1)
        x_path1_half = self.relu(x_path1)
        x_path1 = self.maxpool_downsample_pathGlobal12(x_path1_half)

        x_path1 = self.path1_block3_conv(x_path1)
        x_path1 = self.path1_block3_bn(x_path1)
        x_path1 = self.relu(x_path1)
        x_path1 = self.path1_block3_conv_1(x_path1)
        x_path1 = self.path1_block3_bn_1(x_path1)
        x_path1_quat = self.relu(x_path1)
        x_path1_0 = self.maxpool_downsample_pathGlobal13(x_path1_quat)
        # print(x_path1.shape)

        """path 2"""
        x_path2 = self.path2_block1_conv(x_path2)
        x_path2 = self.path2_block1_bn(x_path2)
        x_path2 = self.relu(x_path2)
        x_path2 = self.path2_block1_conv_1(x_path2)
        x_path2 = self.path2_block1_bn_1(x_path2)
        x_path2 = self.relu(x_path2)
        x_path2 = self.maxpool_downsample_pathGlobal21(x_path2)

        x_path2 = self.path2_block2_conv(x_path2)
        x_path2 = self.path2_block2_bn(x_path2)
        x_path2 = self.relu(x_path2)
        x_path2 = self.path2_block2_conv_1(x_path2)
        x_path2 = self.path2_block2_bn_1(x_path2)
        x_path2_half = self.relu(x_path2)
        x_path2 = self.maxpool_downsample_pathGlobal22(x_path2_half)

        x_path2 = self.path2_block3_conv(x_path2)
        x_path2 = self.path2_block3_bn(x_path2)
        x_path2 = self.relu(x_path2)
        x_path2 = self.path2_block3_conv_1(x_path2)
        x_path2 = self.path2_block3_bn_1(x_path2)
        x_path2_quat = self.relu(x_path2)
        x_path2_0 = self.maxpool_downsample_pathGlobal23(x_path2_quat)
        # print(x_path2_0.size())

        x_path1 = self.path1_block3_NLCross(x_path1_0, x_path2_0)
        x_path1 = self.relu(x_path1)

        x_path2 = self.path2_block3_NLCross(x_path2_0, x_path1_0)
        x_path2 = self.relu(x_path2)


        """path combined"""
        x_pathC1 = torch.cat((x_path1, x_path2_0), 1)
        x_pathC2 = torch.cat((x_path1_0, x_path2), 1)

        x_pathC1 = self.pathC_bn1_1(x_pathC1)
        x_pathC2 = self.pathC_bn1_2(x_pathC2)
        x_pathC1 = self.conv3d_7(x_pathC1)
        x_pathC2 = self.conv3d_7(x_pathC2)
        x = x_pathC1 + x_pathC2
        x = self.relu(x)


        x = self.upConv(x)
        x = self.upConv_bn(x)
        x = self.relu(x)

        x_combined_quat = torch.cat([x_path1_quat,x_path2_quat], dim=1)
        x_combined_quat = self.conv3d_quat_downChannel(x_combined_quat)
        x = torch.cat([x, x_combined_quat], dim=1) #32 channels each
        x = self.conv3d_8(x)
        x = self.conv3d_8_bn(x)
        x = self.relu(x)
        x = self.conv3d_8_1(x)
        x = self.conv3d_8_1_bn(x)
        x = self.relu(x)

        x = self.upConv2(x)
        x_combined_half = torch.cat([x_path1_half,x_path2_half], dim=1)
        x_combined_half = self.conv3d_half_downChannel(x_combined_half)
        x = torch.cat([x, x_combined_half], dim=1) #32 channels each

        x = self.conv3d_9(x)
        x = self.conv3d_9_bn(x)
        x = self.relu(x)
        x = self.conv3d_9_1(x)
        x = self.conv3d_9_1_bn(x)
        x = self.relu(x)


        x = self.flow(x)
        x = F.interpolate(x, scale_factor=2, mode='trilinear',align_corners=True)


        warped = self.transformer(moving_img, x)

        return warped, x
