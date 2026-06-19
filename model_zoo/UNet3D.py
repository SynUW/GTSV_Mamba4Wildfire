"""
Adapted from: 
https://github.com/milesial/Pytorch-UNet/tree/master/unet
https://github.com/jphdotam/Unet3D

using avgpooling for classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs, n_channels=38, n_classes=1, width_multiplier=1, trilinear=True, use_ds_conv=False):
        """A simple 3D Unet, adapted from a 2D Unet from https://github.com/milesial/Pytorch-UNet/tree/master/unet
        Arguments:
          n_channels = number of input channels; 3 for RGB, 1 for grayscale input
          n_classes = number of output channels/classes
          width_multiplier = how much 'wider' your UNet should be compared with a standard UNet
                  default is 1;, meaning 32 -> 64 -> 128 -> 256 -> 512 -> 256 -> 128 -> 64 -> 32
                  higher values increase the number of kernels pay layer, by that factor
          trilinear = use trilinear interpolation to upsample; if false, 3D convtranspose layers will be used instead
          use_ds_conv = if True, we use depthwise-separable convolutional layers. in my experience, this is of little help. This
                  appears to be because with 3D data, the vast vast majority of GPU RAM is the input data/labels, not the params, so little
                  VRAM is saved by using ds_conv, and yet performance suffers."""
        super(Model, self).__init__()
        # 4 层通道，3 次下采样（适配小尺寸如 T=10,H=13,W=13，避免下采样后尺寸为 0）
        _channels = (32, 64, 128, 256)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c * width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d

        self.inc = DoubleConv(n_channels, self.channels[0], conv_type=self.convtype)
        self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down3 = Down(self.channels[2], self.channels[3], conv_type=self.convtype)
        factor = 2 if trilinear else 1
        # Up(in_channels=concat 后通道数, out_channels)；上一级 out 为 channels[i]//factor
        self.up2 = Up(self.channels[3] + self.channels[2], self.channels[2] // factor, trilinear)
        self.up3 = Up(self.channels[2] // factor + self.channels[1], self.channels[1] // factor, trilinear)
        self.up4 = Up(self.channels[1] // factor + self.channels[0], self.channels[0], trilinear)
        # 分类头：全局池化 + 全连接，输出 (B, n_classes)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(self.channels[0], n_classes)

    def forward(self, x, x_mark_enc=None, x_mark_dec=None, mask=None, burn_history=None):
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)           # (B, channels[0], D, H, W)
        x = self.pool(x)               # (B, channels[0], 1, 1, 1)
        x = x.view(B, -1)             # (B, channels[0])
        logits = self.fc(x)           # (B, n_classes)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 3D: (B, C, D, H, W)，对 D/H/W 三维对齐
        diffD = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [
            diffW // 2, diffW - diffW // 2,
            diffH // 2, diffH - diffH // 2,
            diffD // 2, diffD - diffD // 2,
        ])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

if __name__ == "__main__":
    B, T, C, H, W = 16, 10, 38, 13, 13
    unet = Model(n_channels=C, n_classes=1)
    x = torch.randn(B, T, C, H, W)
    y = unet(x)  # (B, n_classes)
    print(y.shape)  # e.g. torch.Size([16, 2])