import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
from source.nn.condinstancenorm import CondInstanceNorm2d
from torch.autograd import Variable
import torchvision.models

class SingleSample(nn.Module):
    def __init__(self, ch_in, ch_out, n_classes=0, kernel_size=3, padding=1, dilation=0):
        super(SingleSample, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.n_clasess = n_classes
        IN = CondInstanceNorm2d if n_classes > 0 else nn.InstanceNorm2d
        self.IN = IN(ch_in, n_classes) if n_classes > 0 else IN(ch_in)
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, padding=padding, dilation=dilation)
        # self.act = activation()

    def forward(self, x, c=None):
        x = self.IN(x, c) if self.n_clasess > 0 else self.IN(x)
        x = self.conv(x)
        return x

class MultiSample(nn.Module):
    def __init__(self, ch_in, ch_out, n_classes=0, cat=True):
        super(MultiSample, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.n_clasess = n_classes
        self.cat = cat

        views = []
        for i in range(5):
            (padding, dilation) = (6*i, 6*i) if i else (1, 1)
            views.append( SingleSample(ch_in, ch_out, n_classes, 3, padding=padding, dilation=dilation) )
        self.views = nn.ModuleList(views)


    def forward(self, x, c=None, attention=None):
        if attention is not None:
            attns = attention
        out = (self.views[0](x, c) if self.n_clasess > 0 else self.views[0](x))
        if attention is not None:
            out = out * attns[0].expand_as(out)
        for i in range(4):
            view = self.views[i + 1](x, c) if self.n_clasess > 0 else self.views[i + 1](x)
            if attention is not None:
                if self.cat:
                    out = torch.cat([out, view * attns[i].expand_as(view)], dim=1)
                else:
                    out = out + view * attns[i].expand_as(out)
            else:
                out = out + view

        return out