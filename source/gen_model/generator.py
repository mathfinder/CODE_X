import torch
import torch.nn as nn
import torch.nn.functional as F
from source.gen_model.resblock import OptimizedBlock, Block



class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, ch=64, n_classes=5, n_repeat=6, activation=F.relu):
        super(Generator, self).__init__()
        self.n_repeat = n_repeat

        self.block1 = OptimizedBlock(3, ch)
        self.block2 = Block(ch    , ch * 2, activation=activation, downsample=True, n_classes=n_classes)
        self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True, n_classes=n_classes)
        blocks = []
        for i in range(n_repeat):
            blocks.append(Block(ch * 4, ch * 4, activation=activation, n_classes=n_classes))
        self.blocks = nn.ModuleList(blocks)
        self.block4 = Block(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
        self.block5 = Block(ch * 2, ch    , activation=activation, upsample=True, n_classes=n_classes)
        self.block6 = Block(ch, ch, activation=activation, upsample=True, n_classes=n_classes)
        # self.block6 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

        views = []
        views.append(nn.Conv2d(ch, 3, 3, padding=1))
        views.append(nn.Conv2d(ch, 3, 3, padding=6, dilation=6))
        views.append(nn.Conv2d(ch, 3, 3, padding=12, dilation=12))
        views.append(nn.Conv2d(ch, 3, 3, padding=18, dilation=18))
        views.append(nn.Conv2d(ch, 3, 3, padding=24, dilation=24))
        self.views = nn.ModuleList(views)
        attns = []
        for i in range(4):
            attns.append(nn.Sequential(nn.Conv2d(ch, 3, 1), nn.Sigmoid()))
        self.attns = nn.ModuleList(attns)

    def forward(self, x, c):
        # out = x
        h = x
        h = self.block1(h)
        h = self.block2(h, c)
        h = self.block3(h, c)
        for i in range(self.n_repeat):
            h = self.blocks[i](h, c)
        h = self.block4(h, c)
        h = self.block5(h, c)
        h = self.block6(h, c)

        out = self.views[0](h)
        for i in range(4):
            out = out + self.views[i+1](h) * self.attns[i](h).expand_as(x)

        return F.tanh(out)


class Generator_small(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, ch=64, n_classes=5, n_repeat=6, activation=F.relu):
        super(Generator_small, self).__init__()
        self.n_repeat = n_repeat

        self.block1 = OptimizedBlock(3, ch)
        self.block2 = Block(ch    , ch * 2, activation=activation, downsample=True, n_classes=n_classes)
        blocks = []
        for i in range(n_repeat):
            blocks.append(Block(ch * 2, ch * 2, activation=activation, n_classes=n_classes))
        self.blocks = nn.ModuleList(blocks)
        self.block4 = Block(ch * 2, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
        self.block5 = Block(ch * 2, ch    , activation=activation, upsample=True, n_classes=n_classes)
        # self.block6 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

        views = []
        views.append(nn.Conv2d(ch, 3, 3, padding=1))
        views.append(nn.Conv2d(ch, 3, 3, padding=6, dilation=6))
        views.append(nn.Conv2d(ch, 3, 3, padding=12, dilation=12))
        views.append(nn.Conv2d(ch, 3, 3, padding=18, dilation=18))
        views.append(nn.Conv2d(ch, 3, 3, padding=24, dilation=24))
        self.views = nn.ModuleList(views)
        attns = []
        for i in range(4):
            attns.append(nn.Sequential(nn.Conv2d(ch, 3, 1), nn.Sigmoid()))
        self.attns = nn.ModuleList(attns)

    def forward(self, x, c):
        # out = x
        h = x
        h = self.block1(h)
        h = self.block2(h, c)
        for i in range(self.n_repeat):
            h = self.blocks[i](h, c)
        h = self.block4(h, c)
        h = self.block5(h, c)

        out = self.views[0](h)
        for i in range(4):
            out = out + self.views[i+1](h) * self.attns[i](h).expand_as(x)

        return F.tanh(out/4.)