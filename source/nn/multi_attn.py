import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.autograd import Variable

class MultiAttn(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(MultiAttn, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        attns = []
        for i in range(4):
            attns.append(nn.Sequential(nn.Conv2d(ch_in, ch_out, 1), nn.Sigmoid()))
        self.attns = nn.ModuleList(attns)

    def forward(self, original, object):

       pass