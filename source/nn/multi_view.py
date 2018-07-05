import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
from source.nn.condinstancenorm import CondInstanceNorm2d
from torch.autograd import Variable

class SingleSample(nn.Module):
    def __init__(self, ch_in, ch_out, n_classes=0, kernel_size=3, padding=1, dilation=0, activation=nn.ReLU):
        super(SingleSample, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.n_clasess = n_classes
        IN = CondInstanceNorm2d if n_classes > 0 else nn.InstanceNorm2d
        self.IN = IN(ch_in, n_classes) if n_classes > 0 else IN(ch_in)
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, padding=padding, dilation=dilation)
        self.act = activation()

    def forward(self, x, c=None):
        x = self.IN(x, c) if self.n_clasess > 0 else self.IN(x)
        x = self.conv(x)
        x = self.act(x)
        return x

class MultiView(nn.Module):
    def __init__(self, ch_in, ch_out, n_classes=0, if_attention=False, activation=nn.ReLU):
        super(MultiView, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.n_clasess = n_classes
        self.if_attention = if_attention
        IN = CondInstanceNorm2d if n_classes > 0 else nn.InstanceNorm2d
        views = []
        for i in range(5):
            (padding, dilation) = (6*i, 6*i) if i else (1, 1)
            views.append( SingleSample(ch_in, ch_out, n_classes, 3, padding=padding, dilation=dilation) )
        self.views = nn.ModuleList(views)

        if if_attention:
            attns = []
            for i in range(4):
                attns.append(nn.Sequential(nn.Conv2d(ch_in, 3, 1), nn.Sigmoid()))
            self.attns = nn.ModuleList(attns)

    def forward(self, x, c=None, attention=None):
        if self.if_attention:
            attns = self.attns
        elif attention is not None:
            attns = attention

        out = self.views[0](x, c) if self.n_clasess > 0 else self.views[0](x)
        for i in range(4):
            view = self.views[i + 1](x, c) if self.n_clasess > 0 else self.views[i + 1](x)
            if self.if_attention:

                out = out + view * attns[i](x).expand_as(out)
            elif attention is not None:
                out = out + view * attns[i].expand_as(out)
            else:
                out = out + view
        out = out / 5.
        return out