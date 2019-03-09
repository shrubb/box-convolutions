import torch
import torch.nn as nn
import torch.nn.functional as F

class ENet(nn.ModuleList):
    def __init__(self, n_classes=19):
        super().__init__([
            Downsampler(3, 16),
            Bottleneck(16, 64, 0.01, downsample=True),

            Bottleneck(64, 64, 0.01),
            Bottleneck(64, 64, 0.01),
            Bottleneck(64, 64, 0.01),
            Bottleneck(64, 64, 0.01),

            Bottleneck(64, 128, 0.1, downsample=True),

            Bottleneck(128, 128, 0.1),
            Bottleneck(128, 128, 0.1, dilation=2),
            Bottleneck(128, 128, 0.1, asymmetric_ksize=5),
            Bottleneck(128, 128, 0.1, dilation=4),
            Bottleneck(128, 128, 0.1),
            Bottleneck(128, 128, 0.1, dilation=8),
            Bottleneck(128, 128, 0.1, asymmetric_ksize=5),
            Bottleneck(128, 128, 0.1, dilation=16),

            Bottleneck(128, 128, 0.1),
            Bottleneck(128, 128, 0.1, dilation=2),
            Bottleneck(128, 128, 0.1, asymmetric_ksize=5),
            Bottleneck(128, 128, 0.1, dilation=4),
            Bottleneck(128, 128, 0.1),
            Bottleneck(128, 128, 0.1, dilation=8),
            Bottleneck(128, 128, 0.1, asymmetric_ksize=5),
            Bottleneck(128, 128, 0.1, dilation=16),

            Upsampler(128, 64),

            Bottleneck(64, 64, 0.1),
            Bottleneck(64, 64, 0.1),

            Upsampler(64, 16),

            Bottleneck(16, 16, 0.1),

            nn.ConvTranspose2d(16, n_classes+1, (2,2), (2,2))])

    def forward(self, x):
        max_indices_stack = []

        for module in self:
            if isinstance(module, Upsampler):
                x = module(x, max_indices_stack.pop())
            else:
                x = module(x)

            if type(x) is tuple: # then it was a downsampling bottleneck block
                x, max_indices = x
                max_indices_stack.append(max_indices)

        return x

class BoxENet(ENet):
    def __init__(self, n_classes=19, max_input_h=512, max_input_w=1024):
        h, w = max_input_h, max_input_w # shorten names for convenience

        nn.ModuleList.__init__(self, [
            Downsampler(3, 16),
            Bottleneck(16, 64, 0.01, downsample=True),

            Bottleneck(64, 64, 0.01),
            BottleneckBoxConv(64, 4, h // 4, w // 4, 0.15),
            Bottleneck(64, 64, 0.01),
            BottleneckBoxConv(64, 4, h // 4, w // 4, 0.15),

            Bottleneck(64, 128, 0.1, downsample=True),

            Bottleneck(128, 128, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.25),
            Bottleneck(128, 128, 0.1, asymmetric_ksize=5),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.25),
            Bottleneck(128, 128, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.25),
            Bottleneck(128, 128, 0.1, asymmetric_ksize=5),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.25),

            Bottleneck(128, 128, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.25),
            Bottleneck(128, 128, 0.1, asymmetric_ksize=5),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.25),
            Bottleneck(128, 128, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.25),
            Bottleneck(128, 128, 0.1, asymmetric_ksize=5),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.25),

            Upsampler(128, 64),

            Bottleneck(64, 64, 0.1),
            BottleneckBoxConv(64, 4, h // 4, w // 4, 0.1),

            Upsampler(64, 16),

            BottleneckBoxConv(16, 2, h // 2, w // 2, 0.1),

            nn.ConvTranspose2d(16, n_classes+1, (2,2), (2,2))])

class BoxOnlyENet(ENet):
    def __init__(self, n_classes=19, max_input_h=512, max_input_w=1024):
        h, w = max_input_h, max_input_w # shorten names for convenience

        nn.ModuleList.__init__(self, [
            Downsampler(3, 16),
            Bottleneck(16, 64, 0.01, downsample=True),

            BottleneckBoxConv(64, 4, h // 4, w // 4, 0.1),
            BottleneckBoxConv(64, 4, h // 4, w // 4, 0.1),
            BottleneckBoxConv(64, 4, h // 4, w // 4, 0.1),
            BottleneckBoxConv(64, 4, h // 4, w // 4, 0.1),

            Bottleneck(64, 128, 0.1, downsample=True),

            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),

            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),
            BottleneckBoxConv(128, 4, h // 8, w // 8, 0.1),

            Upsampler(128, 64),

            BottleneckBoxConv(64, 4, h // 4, w // 4, 0.1),
            BottleneckBoxConv(64, 4, h // 4, w // 4, 0.1),

            Upsampler(64, 16),

            BottleneckBoxConv(16, 4, h // 2, w // 2, 0.1),

            nn.ConvTranspose2d(16, n_classes+1, (2,2), (2,2))])


class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        bt_channels = in_channels // 4

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, bt_channels, (1,1), bias=False),
            nn.BatchNorm2d(bt_channels, 1e-3),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(bt_channels, bt_channels, (3,3), 2, 1, 1),
            nn.BatchNorm2d(bt_channels, 1e-3),
            nn.ReLU(True),

            nn.Conv2d(bt_channels, out_channels, (1,1), bias=False),
            nn.BatchNorm2d(out_channels, 1e-3))

        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1,1), bias=False),
            nn.BatchNorm2d(out_channels, 1e-3))

    def forward(self, x, max_indices):
        x_skip_connection = self.skip_connection(x)
        x_skip_connection = F.max_unpool2d(x_skip_connection, max_indices, (2,2))

        return F.relu(x_skip_connection + self.main_branch(x), inplace=True)

class Downsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels-in_channels, (3,3), 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, 1e-3)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = torch.cat([F.max_pool2d(x, (2,2)), self.conv(x)], 1)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Bottleneck(nn.Module):
    def __init__(
        self, in_channels, out_channels, dropout_prob=0.0, downsample=False,
        asymmetric_ksize=None, dilation=1, use_prelu=True):

        super().__init__()
        bt_channels = in_channels // 4
        self.downsample = downsample
        self.channels_to_pad = out_channels-in_channels

        input_stride = 2 if downsample else 1

        main_branch = [
            nn.Conv2d(in_channels, bt_channels, input_stride, input_stride, bias=False),
            nn.BatchNorm2d(bt_channels, 1e-3),
            nn.PReLU(bt_channels) if use_prelu else nn.ReLU(True)
        ]
       
        if asymmetric_ksize is None:
            main_branch += [
                nn.Conv2d(bt_channels, bt_channels, (3,3), 1, dilation, dilation)
            ]
        else:
            assert type(asymmetric_ksize) is int
            ksize, padding = asymmetric_ksize, (asymmetric_ksize-1) // 2
            main_branch += [
                nn.Conv2d(bt_channels, bt_channels, (ksize,1), 1, (padding,0), bias=False),
                nn.Conv2d(bt_channels, bt_channels, (1,ksize), 1, (0,padding))
            ]
       
        main_branch += [
            nn.BatchNorm2d(bt_channels, 1e-3),
            nn.PReLU(bt_channels) if use_prelu else nn.ReLU(True),
            nn.Conv2d(bt_channels, out_channels, (1,1), bias=False),
            nn.BatchNorm2d(out_channels, 1e-3),
            nn.Dropout2d(dropout_prob)
        ]

        self.main_branch = nn.Sequential(*main_branch)        
        self.output_activation = nn.PReLU(out_channels) if use_prelu else nn.ReLU(True)

    def forward(self, x):
        if self.downsample:
            x_skip_connection, max_indices = F.max_pool2d(x, (2,2), return_indices=True)
        else:
            x_skip_connection = x

        if self.channels_to_pad > 0:
            x_skip_connection = F.pad(x_skip_connection, (0,0, 0,0, 0,self.channels_to_pad))

        x = self.output_activation(x_skip_connection + self.main_branch(x))
        
        if self.downsample:
            return x, max_indices
        else:
            return x

from box_convolution import BoxConv2d

class BottleneckBoxConv(nn.Module):
    def __init__(self, in_channels, num_boxes, max_input_h, max_input_w, dropout_prob=0.0):
        super().__init__()
        assert in_channels % num_boxes == 0
        bt_channels = in_channels // num_boxes # bottleneck channels

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, bt_channels, (1,1), bias=False),
            nn.BatchNorm2d(bt_channels),
            nn.ReLU(True),
            
            # BEHOLD:
            BoxConv2d(
                bt_channels, num_boxes, max_input_h, max_input_w,
                reparametrization_factor=1.5625),

            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(dropout_prob))

    def forward(self, x):
        return F.relu(x + self.main_branch(x), inplace=True)

