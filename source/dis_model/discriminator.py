import torch
import torch.nn as nn
import torch.nn.functional as F
from source.dis_model.resblock import Block, OptimizedBlock, SNBlock, SNOptimizedBlock
from source.modules.sn_linear import SNLinear
from source.modules.sn_conv2d import SNConv2d
from source.modules.sn_embedding import SNEmbedding
class Discriminator(nn.Module):
    def __init__(self, ch=64, n_classes=0, activation=F.relu):
        super(Discriminator, self).__init__()
        self.activation = activation

        self.block1 = OptimizedBlock(3, ch)
        self.block2 = Block(ch, ch * 2, activation=activation, downsample=True)
        self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
        self.block4 = Block(ch * 4, ch * 8, activation=activation, downsample=True)
        self.block5 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
        self.block6 = Block(ch * 16, ch * 16, activation=activation, downsample=True)

        self.conv1 = nn.Conv2d(ch * 16, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l_cls = nn.Linear(ch * 16, n_classes)
        self.l_reg = nn.Linear(ch * 16, 1)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)

        rf  = F.sigmoid(self.conv1(h))
        h   = torch.sum(torch.sum(h, 3), 2)
        cls = self.l_cls(h)
        reg = self.l_reg(h)

        return rf, cls, reg

class SNDiscriminator(nn.Module):
    def __init__(self, ch=64, n_classes=0, activation=F.relu):
        super(SNDiscriminator, self).__init__()
        self.activation = activation

        self.block1 = SNOptimizedBlock(3, ch)
        self.block2 = SNBlock(ch, ch * 2, activation=activation, downsample=True)
        self.block3 = SNBlock(ch * 2, ch * 4, activation=activation, downsample=True)
        self.block4 = SNBlock(ch * 4, ch * 8, activation=activation, downsample=True)
        self.block5 = SNBlock(ch * 8, ch * 16, activation=activation, downsample=True)
        self.block6 = SNBlock(ch * 16, ch * 16, activation=activation, downsample=True)

        self.conv1 = SNConv2d(ch * 16, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l_cls = SNLinear(ch * 16, n_classes)
        self.l_reg = SNLinear(ch * 16, 1)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        # h = self.activation(h)

        rf  = F.sigmoid(self.conv1(h))
        h   = torch.sum(torch.sum(h, 3), 2)
        cls = self.l_cls(h)
        reg = self.l_reg(h)

        return rf, cls, reg

class SNResNetProjectionDiscriminator(nn.Module):
    def __init__(self, ch=64, ch_in=3, n_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.activation = activation

        self.block1 = OptimizedBlock(ch_in, ch)
        self.block2 = Block(ch, ch * 2, activation=activation, downsample=True)
        self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
        self.block4 = Block(ch * 4, ch * 8, activation=activation, downsample=True)
        self.block5 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
        self.block6 = Block(ch * 16, ch * 16, activation=activation, downsample=False)
        self.l7 = SNLinear(ch * 16, 1)
        self.l7_reg = SNLinear(ch * 16, 1)
        if n_classes > 0:
            self.l_y = SNEmbedding(n_classes, ch * 16)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        h = torch.sum(torch.sum(h, 3), 2)  # Global pooling
        output = self.l7(h)
        reg = self.l7_reg(h)
        if y is not None:
            w_y = self.l_y(y)
            output = output + torch.sum(w_y * h, dim=1, keepdim=True)
        return output, reg