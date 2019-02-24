import torch
import torch.nn as nn
import torch.nn.functional as F

def ERFNet(n_classes=19):
    return nn.Sequential(
        Downsampler( 3, 16, 0.0 ),
        Downsampler(16, 64, 0.03),

        NonBottleneck1D(64, 0.03),
        NonBottleneck1D(64, 0.03),
        NonBottleneck1D(64, 0.03),
        NonBottleneck1D(64, 0.03),
        NonBottleneck1D(64, 0.03),

        Downsampler(64, 128, 0.3),

        NonBottleneck1D(128, 0.3,  2),
        NonBottleneck1D(128, 0.3,  4),
        NonBottleneck1D(128, 0.3,  8),
        NonBottleneck1D(128, 0.3, 16),
        NonBottleneck1D(128, 0.3,  2),
        NonBottleneck1D(128, 0.3,  4),
        NonBottleneck1D(128, 0.3,  8),
        NonBottleneck1D(128, 0.3, 16),

        Upsampler(128, 64),

        NonBottleneck1D(64),
        NonBottleneck1D(64),

        Upsampler(64, 16),

        NonBottleneck1D(16),
        NonBottleneck1D(16),

        nn.ConvTranspose2d(16, n_classes+1, (3,3), 2, 1, 1))

def BoxERFNet(n_classes=19, max_input_h=512, max_input_w=1024):
    h, w = max_input_h, max_input_w # shorten names for convenience

    return nn.Sequential(
        Downsampler( 3, 16, 0.0 ),
        Downsampler(16, 64, 0.03),

        NonBottleneck1D(64, 0.03),
        BottleneckBoxConv(64, 4, h // 4, w // 4, 0.03),

        Downsampler(64, 128, 0.3),

        NonBottleneck1D(128, 0.3, 2),
        BottleneckBoxConv(128, 4, h // 8, w // 8, 0.3),
        NonBottleneck1D(128, 0.3, 4),

        BottleneckBoxConv(128, 4, h // 8, w // 8, 0.3),

        NonBottleneck1D(128, 0.3, 2),
        BottleneckBoxConv(128, 4, h // 8, w // 8, 0.3),
        NonBottleneck1D(128, 0.3, 4),
        BottleneckBoxConv(128, 4, h // 8, w // 8, 0.3),

        Upsampler(128, 64),

        NonBottleneck1D(64),

        Upsampler(64, 16),

        NonBottleneck1D(16),

        nn.ConvTranspose2d(16, n_classes+1, (3,3), 2, 1, 1))

def Upsampler(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, (3,3), 2, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))

class Downsampler(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels-in_channels, (3,3), 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        x = torch.cat([F.max_pool2d(x, (2,2)), self.conv(x)], 1)
        x = self.bn(x)
        x = self.dropout(x)
        x = F.relu(x, inplace=True)
        return x

class NonBottleneck1D(nn.Module):
    def __init__(self, in_channels, dropout_prob=0.0, dilation=1):
        super().__init__()
        dil = dilation # shorten the name for convenience

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3,1), 1, (1,0), bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, (1,3), 1, (0,1), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),

            nn.Conv2d(in_channels, in_channels, (3,1), 1, (dil,0), (dil,dil), bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, (1,3), 1, (0,dil), (dil,dil), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(dropout_prob))

    def forward(self, x):
        return F.relu(x + self.main_branch(x), inplace=True)

from box_convolution import BoxConv2d

class BottleneckBoxConv(nn.Module):
    def __init__(self, in_channels, num_boxes, max_input_h, max_input_w, dropout_prob=0.0):
        super().__init__()
        assert in_channels % num_boxes == 0
        bt_channels = in_channels // num_boxes # bottleneck channels

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, bt_channels, (1,1), bias=False),
            nn.BatchNorm2d(bt_channels),
            
            # BEHOLD:
            BoxConv2d(bt_channels, num_boxes, max_input_h, max_input_w),

            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(dropout_prob))

    def forward(self, x):
        return F.relu(x + self.main_branch(x), inplace=True)
