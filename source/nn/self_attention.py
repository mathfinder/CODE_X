import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SelfAttention, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.f = nn.Conv2d(ch_in, ch_out, 1)
        self.g = nn.Conv2d(ch_in, ch_out, 1)
        self.h = nn.Conv2d(ch_in, ch_in, 1)
        # self.gamma = Parameter(torch.Tensor(1).normal_(mean=0, std=0.02))
        # self.gamma = Parameter(torch.Tensor(1).fill_(0))

    def forward(self, x):
        b, c, H, W = x.size()
        f = self.f(x).reshape(b, self.ch_out, -1)
        g = self.g(x).reshape(b, self.ch_out, -1)
        h = self.h(x).reshape(b, self.ch_in , -1)
        # todo
        # s = torch.bmm(f.transpose(1, 2), g)
        # beta = F.softmax(s, dim=1).transpose(1, 2)
        # o = torch.bmm(beta, h.transpose(1, 2)).transpose(1, 2)

        s = torch.bmm(f.transpose(1, 2), g)
        beta = F.softmax(s, dim=1)
        o = torch.bmm(h, beta).view(*x.size())
        # o = self.gamma * torch.bmm(h, beta).view(*x.size())
        return o