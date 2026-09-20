import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import cv2

from config import Config as cfg
from layers import *
import utils


class AffineSteam(nn.Module):
    def __init__(self, nc, nf=32):
        super(AffineSteam, self).__init__()
        self.down1 = ConvRelu(nc, nf, 4, 2, 1)  # 64 -> 32
        self.down2 = ConvRelu(nf, nf*2, 4, 2, 1)  # 32 -> 16
        self.down3 = ConvRelu(nf*2, nf*2, 4, 2, 1) # 16  -> 8
        self.down4 = ConvRelu(nf*2, nf*4, 4, 2, 1) # 8  -> 4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_w = nn.Linear(nf*4, 2*2, bias=False)
        self.fc_b = nn.Linear(nf*4, 1*2, bias=False)

    def forward(self, x):
        size = x.size()
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.pool(x)
        x = x.view(size[0], -1)

        W = self.fc_w(x).view(-1, 2, 2)
        b = self.fc_b(x).view(-1, 2)
        grid = self.aff_flow(W, b, size[2], size[3])
        grid = grid.permute(0, 3, 1, 2)
        return W, b, grid

    def aff_flow(self, W, b, xs, ys):
        b = b.view(-1, 1, 1, 2)
        xr = torch.arange(-(xs-1)/2., xs/2., 1.0, dtype=torch.float32).view(1, -1, 1, 1)
        yr = torch.arange(-(ys-1)/2., ys/2., 1.0, dtype=torch.float32).view(1, 1, -1, 1)

        wx = W[:, :, 0].view(-1, 1, 1, 2)
        wy = W[:, :, 1].view(-1, 1, 1, 2)

        return xr*wx + yr*wy + b

class Encoder(nn.Module):
    def __init__(self, nc=3, nf=32, bottle=512, logvar_max=2, logvar_min=-10):
        super().__init__()
        self.bottle = bottle
        self.logvar_max = logvar_max
        self.logvar_min = logvar_min

        self.down1 = nn.Sequential(
                        ConvRelu(nc, nf, 4, 2, 1), # 128 -> 64
                        ConvRelu(nf, nf))
        self.down2 = nn.Sequential(
                        ConvRelu(nf, nf*2, 4, 2, 1),
                        ConvRelu(nf*2, nf*2))  # 64 -> 32
        self.down3 = nn.Sequential(
                        ConvRelu(nf*2, nf*2, 4, 2, 1),
                        ConvRelu(nf*2, nf*2))  # 32 -> 16
        self.down4 = nn.Sequential(
                        ConvRelu(nf*2, nf*4, 4, 2, 1),
                        ConvRelu(nf*4, nf*4)) # 16  -> 8
        self.down5 = nn.Sequential(
                        ConvRelu(nf*4, nf*4, 4, 2, 1),
                        ConvRelu(nf*4, nf*4)) # 8  -> 4

        # Add adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        # Add adaptive pooling for skip connections to ensure consistent spatial dimensions
        self.adaptive_pool_8 = nn.AdaptiveAvgPool3d((8, 8, 8))  # for enc[-2]
        self.adaptive_pool_16 = nn.AdaptiveAvgPool3d((16, 16, 16))  # for enc[-3]
        self.adaptive_pool_32 = nn.AdaptiveAvgPool3d((32, 32, 32))  # for enc[-4]
        self.adaptive_pool_64 = nn.AdaptiveAvgPool3d((64, 64, 64))  # for enc[-5]
        self.adaptive_pool_128 = nn.AdaptiveAvgPool3d((128, 128, 128))  # for enc[0] (input level)
        self.out = ConvRelu(nf*4, (bottle)*2, 4, 1, 0) # 4x4x4 -> 1x1x1

    def dist_multivar_normal(self, mu, logvar):
        var = logvar.exp() # 协方差对角 var >=0
        cov_matrix = torch.diag_embed(var) # 生成正定协方差（对角）矩阵
        dist = torch.distributions.MultivariateNormal(mu, cov_matrix)
        return dist

    def forward(self, x):
        batch = x.size(0)
        enc = [x] # 2, 128
        x = self.down1(x)
        enc.append(x) # 16, 64
        x = self.down2(x)
        enc.append(x) # 64, 32
        x = self.down3(x)
        enc.append(x) # 32, 16
        x = self.down4(x)
        enc.append(x) # 16, 8
        x = self.down5(x)
        enc.append(x) # 8, 4 (or variable size)

        # Ensure consistent spatial size before final conv
        x = self.adaptive_pool(x)  # -> 4x4x4
        
        # Apply adaptive pooling to skip connections to ensure consistent spatial dimensions
        # Ensure input level is exactly 128x128x128
        enc[0] = self.adaptive_pool_128(enc[0])  # 128x128x128
        # enc[-1] corresponds to the 4x4x4 level (x after down5) - already handled by adaptive_pool
        enc[-1] = self.adaptive_pool(enc[-1])  # 4x4x4
        enc[-2] = self.adaptive_pool_8(enc[-2])  # 8x8x8
        enc[-3] = self.adaptive_pool_16(enc[-3])  # 16x16x16
        enc[-4] = self.adaptive_pool_32(enc[-4])  # 32x32x32
        enc[-5] = self.adaptive_pool_64(enc[-5])  # 64x64x64
        
        x = self.out(x) # 4x4x4 -> 1x1x1, bottle*2 channels
        x = x.view(batch, -1)
        mu, logvar = x.chunk(2, dim=1)

        logvar = torch.tanh(logvar)
        logvar = self.logvar_min + 0.5 * (
            self.logvar_max - self.logvar_min
        ) * (logvar + 1)

        dist = self.dist_multivar_normal(mu, logvar)

        return dist, enc


class Decoder(nn.Module):
    def __init__(self, nf, bottle):
        super().__init__()
        self.bottle = bottle
        self.up1 = UpSampling(bottle, nf*4, 4)
        self.up2 = UpSampling(nf*4 + nf*4, nf*4)
        self.up3 = UpSampling(nf*4 + nf*4, nf*4)
        self.up4 = UpSampling(nf*4 + nf*2, nf*2)
        self.up5 = UpSampling(nf*2 + nf*2, nf*1)
        self.up6 = UpSampling(nf*1 + nf*1, nf*1)

        self.out = nn.Sequential(
            Conv(nf, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x, enc):
        batch = x.size(0)
        x = x.view(batch, self.bottle, 1, 1, 1)
        x = self.up1(x) # 4
        x = torch.cat([x, enc[-1]], dim=1)
        x = self.up2(x) # 8
        x = torch.cat([x, enc[-2]], dim=1)
        x = self.up3(x) # 16
        x = torch.cat([x, enc[-3]], dim=1)
        x = self.up4(x) # 32
        x = torch.cat([x, enc[-4]], dim=1)
        x = self.up5(x) # 64
        x = torch.cat([x, enc[-5]], dim=1)
        x = self.up6(x) # 128

        x = self.out(x)

        # return torch.clamp(x, -3, 3)
        return x


class VAE(nn.Module):
    def __init__(self, cfg):
        super(VAE, self).__init__()
        self.enc = Encoder(cfg.STATE_CHANNEL, cfg.NF, cfg.BOTTLE)
        self.dec = Decoder(cfg.NF, cfg.BOTTLE)

    def forward(self, x):
        dist, enc = self.enc(x)
        latent = dist.mean
        field = self.dec(latent, enc)
        return field

class Critic(nn.Module):
    def __init__(self, nc, nf=32, bottle=512):
        super(Critic, self).__init__()
        # bottle += 12
        self.conv1 = ConvRelu(nc, nf*1, 4, 2, 1) # 128 -> 64
        self.conv2 = ConvRelu(nf*1, nf*2, 4, 2, 1) # 64 -> 32
        self.conv3 = ConvRelu(nf*2, nf*2, 4, 2, 1) # 32 -> 16
        self.conv4 = ConvRelu(nf*2, nf*4, 4, 2, 1) # 16 > 8
        self.conv5 = ConvRelu(nf*4, nf*4, 4, 2, 1) # 8 -> 4
        # Add adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.conv7 = Conv(nf*4, bottle, 4, 1, 0) # 1
        self.fc = nn.Linear(bottle*2, 1)

    def forward(self, x, latent):
        batch = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # Ensure consistent spatial size before final conv
        x = self.adaptive_pool(x)  # -> 4x4x4
        x = self.conv7(x).view(batch, -1)
        x = torch.cat([x, latent], dim=1)
        x = F.leaky_relu(x, 0.1)
        # print(x.shape, latent.shape)
        value = self.fc(x)

        return value



class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()
        if isinstance(size, int):
            size = (size, size, size)
        self.size = size
        self.mode = mode
        # Grid will be created dynamically based on flow field size

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        # Create grid dynamically based on flow field size
        shape = flow.shape[2:]
        
        # Create sampling grid with the same spatial dimensions as flow
        vectors = [torch.arange(0, s, device=flow.device) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor).to(flow.device)
        
        # Expand grid to match batch size
        if grid.shape[0] != flow.shape[0]:
            grid = grid.repeat(flow.shape[0], 1, 1, 1, 1)
        
        # print(grid.shape, flow.shape)
        new_locs = grid + flow

        # Resize source image if its spatial dimensions don't match flow field
        src_shape = src.shape[2:]
        if src_shape != shape:
            src = F.interpolate(src, size=shape, mode='trilinear', align_corners=True)

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)



class OldSpatialTransformer(nn.Module):
    def __init__(self, device=None):
        self.device = device
        super(OldSpatialTransformer, self).__init__()

    def meshgrid(self, height, width):
        x_t = torch.matmul(torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width -1.0, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]))

        x_t = x_t.expand([height, width])
        y_t = y_t.expand([height, width])
        x_t = x_t.to(self.device)
        y_t = y_t.to(self.device)

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        x = x.to(self.device)
        return torch.squeeze(torch.reshape(x, (-1, 1)))


    def interpolate(self, im, x, y):

        im = F.pad(im, (0,0,1,1,1,1,0,0))

        batch_size, height, width, channels = im.shape

        batch_size, out_height, out_width = x.shape

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        x = x + 1
        y = y + 1

        max_x = width - 1
        max_y = height - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width*height
        base = self.repeat(torch.arange(0, batch_size)*dim1, out_height*out_width)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.float()
        dim, _ = idx_a.transpose(1,0).shape
        Ia = torch.gather(im_flat, 0, idx_a.transpose(1,0).expand(dim, channels))
        Ib = torch.gather(im_flat, 0, idx_b.transpose(1,0).expand(dim, channels))
        Ic = torch.gather(im_flat, 0, idx_c.transpose(1,0).expand(dim, channels))
        Id = torch.gather(im_flat, 0, idx_d.transpose(1,0).expand(dim, channels))

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()

        dx = x1_f - x
        dy = y1_f - y

        wa = (dx * dy).transpose(1,0)
        wb = (dx * (1-dy)).transpose(1,0)
        wc = ((1-dx) * dy).transpose(1,0)
        wd = ((1-dx) * (1-dy)).transpose(1,0)

        output = torch.sum(torch.squeeze(torch.stack([wa*Ia, wb*Ib, wc*Ic, wd*Id], dim=1)), 1)
        output = torch.reshape(output, [-1, out_height, out_width, channels])
        return output.permute(0, 3, 1, 2)

    def forward(self, moving_image, deformation_matrix):
        """moving_image: [h, c, h, w]
           deformation_matrix: [h, c, h, w]
        """
        dx = deformation_matrix[:, 0, :, :]
        dy = deformation_matrix[:, 1, :, :]
        moving_image = moving_image.permute(0,2,3,1)

        batch_size, height, width = dx.shape

        x_mesh, y_mesh = self.meshgrid(height, width)

        x_mesh = x_mesh.expand([batch_size, height, width])
        y_mesh = y_mesh.expand([batch_size, height, width])
        x_new = dx + x_mesh
        y_new = dy + y_mesh
        return self.interpolate(moving_image, x_new, y_new)



























