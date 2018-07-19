import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Conv2d
from source.nn.condinstancenorm import CondInstanceNorm2d

def _downsample(x):
    # todo
    return F.avg_pool2d(x, 2)

def _upsample(x):
    # todo
    h, w = x.shape[2:]
    return F.upsample(x, size=(h * 2, w * 2))

def upsample_conv(x, conv):
    return conv(_upsample(x))

class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation
        self.c1 = Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c2 = Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c_sc = Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, downsample=False, n_classes=0):
        super(Block, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or upsample or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes

        self.c1 = Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        if n_classes > 0:
            self.b1 = CondInstanceNorm2d(in_channels, n_classes)
            self.b2 = CondInstanceNorm2d(hidden_channels, n_classes)
        else:
            self.b1 = nn.InstanceNorm2d(in_channels)
            self.b2 = nn.InstanceNorm2d(hidden_channels)
        if self.learnable_sc:
            self.c_sc = Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def residual(self, x, y=None):
        h = x
        h = self.b1(h, y) if y is not None else self.b1(h)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y) if y is not None else self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x, y=None):
        return self.residual(x, y) + self.shortcut(x)

class EnhanceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, downsample=False, n_classes=0):
        super(Block, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or upsample or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        if upsample:
            self.c1 = nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad, stride=2, output_padding=1)
        else:
            self.c1 = Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        if n_classes > 0:
            self.b1 = CondInstanceNorm2d(in_channels, n_classes)
            self.b2 = CondInstanceNorm2d(hidden_channels, n_classes)
        else:
            self.b1 = nn.InstanceNorm2d(in_channels)
            self.b2 = nn.InstanceNorm2d(hidden_channels)
        if self.learnable_sc:
            if upsample:
                self.c_sc = nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=1, stride=2,
                                             output_padding=1)
            else:
                self.c_sc = Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def residual(self, x, y=None):
        h = x
        h = self.b1(h, y) if y is not None else self.b1(h)
        h = self.activation(h)
        h = self.c1(h)
        h = self.b2(h, y) if y is not None else self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x, y=None):
        return self.residual(x, y) + self.shortcut(x)