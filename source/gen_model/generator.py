import torch
import torch.nn as nn
import torch.nn.functional as F
from source.gen_model.resblock import OptimizedBlock, Block
from source.nn.self_attention import SelfAttention
from source.nn.multi_view import MultiView
from source.nn.condinstancenorm import CondInstanceNorm2d


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
    def __init__(self, ch=64, n_classes=5, n_repeat=6, activation=F.relu, identity=True, self_attention=False):
        super(Generator_small, self).__init__()
        self.n_repeat = n_repeat
        self.identity = identity
        self.self_attention = self_attention
        self.block1 = OptimizedBlock(3, ch)
        self.block2 = Block(ch    , ch * 2, activation=activation, downsample=True, n_classes=n_classes)
        blocks = []
        for i in range(n_repeat):
            blocks.append(Block(ch * 2, ch * 2, activation=activation, n_classes=n_classes))

        self.blocks = nn.ModuleList(blocks)
        self.block4 = Block(ch * 2, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
        self.block5 = Block(ch * 2, ch    , activation=activation, upsample=True, n_classes=n_classes)
        # self.block6 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        if self_attention:
            self.SelfAttention = SelfAttention(ch * 2, ch * 2 // 8)
        views = []
        views.append(nn.Conv2d(ch, 3, 3, padding=1))
        views.append(nn.Conv2d(ch, 3, 3, padding=6, dilation=6))
        views.append(nn.Conv2d(ch, 3, 3, padding=12, dilation=12))
        views.append(nn.Conv2d(ch, 3, 3, padding=18, dilation=18))
        views.append(nn.Conv2d(ch, 3, 3, padding=24, dilation=24))
        self.views = nn.ModuleList(views)
        attns = []
        for i in range(4 + identity):
            attns.append(nn.Sequential(nn.Conv2d(ch, 3, 1), nn.Sigmoid()))
        self.attns = nn.ModuleList(attns)

    def forward(self, x, c):
        # out = x
        h = x
        h = self.block1(h)
        h = self.block2(h, c)
        for i in range(self.n_repeat):
            h = self.blocks[i](h, c)
        if self.self_attention:
            h = self.SelfAttention(h) + h
        h = self.block4(h, c)
        h = self.block5(h, c)

        out = self.views[0](h)
        for i in range(4):
            out = out + self.views[i+1](h) * self.attns[i](h).expand_as(x)
        if self.identity:
            out = (out + x * self.attns[4](h).expand_as(x))/5.
        else:
            out = out/4.
        return F.tanh(out)

class Generator_small_SelfAttention(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, ch=64, n_classes=5, n_repeat=6, activation=F.relu):
        super(Generator_small_SelfAttention, self).__init__()
        self.n_repeat = n_repeat

        self.block1 = OptimizedBlock(3, ch)
        self.block2 = Block(ch    , ch * 2, activation=activation, downsample=True, n_classes=n_classes)
        blocks = []
        for i in range(n_repeat):
            blocks.append(Block(ch * 2, ch * 2, activation=activation, n_classes=n_classes))
        self.blocks = nn.ModuleList(blocks)

        self.SelfAttention = SelfAttention(ch * 2, ch * 2 // 8)

        self.block4 = Block(ch * 2, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
        self.block5 = Block(ch * 2, ch    , activation=activation, upsample=True, n_classes=n_classes)
        self.block6 = Block(ch    ,  3    , activation=activation, n_classes=n_classes)

    def forward(self, x, c):
        # out = x
        h = x
        h = self.block1(h)
        h = self.block2(h, c)
        for i in range(self.n_repeat):
            h = self.blocks[i](h, c)
        h = self.SelfAttention(h) + h
        h = self.block4(h, c)
        h = self.block5(h, c)
        out = self.block6(h, c)

        return F.tanh(out)

class Generator_small_mutiviewsample(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, ch=64, n_classes=5, n_repeat=6, activation=F.relu):
        super(Generator_small_mutiviewsample, self).__init__()
        self.n_repeat = n_repeat

        self.multiViewSample = MultiView(ch_in=3, ch_out=ch, n_classes=n_classes)

        self.block1 = OptimizedBlock(3, ch)
        self.block2 = Block(ch    , ch * 2, activation=activation, downsample=True, n_classes=n_classes)
        blocks = []
        for i in range(n_repeat):
            blocks.append(Block(ch * 2, ch * 2, activation=activation, n_classes=n_classes))
        self.blocks = nn.ModuleList(blocks)

        self.SelfAttention = SelfAttention(ch * 2, ch * 2 // 8)

        self.block4 = Block(ch * 2, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
        self.block5 = Block(ch * 2, ch    , activation=activation, upsample=True, n_classes=n_classes)

        attns = []
        for i in range(4):
            attns.append(nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid()))
        self.attns = nn.ModuleList(attns)

        # self.block6 = Block(ch    ,  3    , activation=activation, n_classes=n_classes)
        self.block6 = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, x, c):
        # out = x

        for i in range(4):
            sample = self.attns
        h = x
        h = self.block1(h)
        h = self.block2(h, c)
        for i in range(self.n_repeat):
            h = self.blocks[i](h, c)
        h = self.SelfAttention(h) + h
        h = self.block4(h, c)
        h = self.block5(h, c)
        attns = []
        for i in range(4):
            attns.append(self.attns[i](h))

        h = self.multiViewSample(x, c, attns) + h
        out = self.block6(h)

        return F.tanh(out)


class Generator_FNT(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, ch=64, n_classes=5, n_repeat=6, activation=F.relu):
        super(Generator_FNT, self).__init__()
        self.n_repeat = n_repeat
        self.activation=activation
        self.c1 = nn.Conv2d(3, ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.i1 = CondInstanceNorm2d(ch, n_classes)

        # Down-Sampling
        self.c2 = nn.Conv2d(ch, ch*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.i2 = CondInstanceNorm2d(ch*2, n_classes)

        self.c3 = nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.i3 = CondInstanceNorm2d(ch * 4, n_classes * 4)

        # Bottleneck
        blocks = []
        for i in range(n_repeat):
            blocks.append(Block(ch * 4, ch * 4, activation=activation, n_classes=n_classes))
        self.blocks = nn.ModuleList(blocks)

        # Up-Sampling
        self.c4 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.i4 = CondInstanceNorm2d(ch * 2, n_classes)

        self.c5 = nn.ConvTranspose2d(ch * 2, ch    , kernel_size=4, stride=2, padding=1, bias=False)
        self.i5 = CondInstanceNorm2d(ch, n_classes)

        self.c6 = nn.Conv2d(ch, 3, kernel_size=7, stride=1, padding=3, bias=False)


    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        h = x
        h = self.c1(h)
        h = self.activation(self.i1(h, c))

        h = self.c2(h)
        h = self.activation(self.i2(h, c))

        h = self.c3(h)
        h = self.activation(self.i3(h, c))

        for i in range(self.n_repeat):
            h = self.blocks[i](h, c)

        h = self.c4(h)
        h = self.activation(self.i4(h, c))

        h = self.c5(h)
        h = self.activation(self.i5(h, c))

        out = F.tanh(self.c6(h))

        return out

class Generator_FNT_mutiviewsample(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, ch=64, n_classes=5, n_repeat=6, activation=F.relu):
        super(Generator_FNT_mutiviewsample, self).__init__()
        self.n_repeat = n_repeat
        self.activation=activation

        self.multiViewSample = MultiView(ch_in=3, ch_out=ch, n_classes=n_classes)

        self.c1 = nn.Conv2d(3, ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.i1 = CondInstanceNorm2d(ch, n_classes)

        # Down-Sampling
        self.c2 = nn.Conv2d(ch, ch*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.i2 = CondInstanceNorm2d(ch*2, n_classes)

        self.c3 = nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.i3 = CondInstanceNorm2d(ch * 4, n_classes * 4)

        # Bottleneck
        blocks = []
        for i in range(n_repeat):
            blocks.append(Block(ch * 4, ch * 4, activation=activation, n_classes=n_classes))
        self.blocks = nn.ModuleList(blocks)

        # Up-Sampling
        self.c4 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.i4 = CondInstanceNorm2d(ch * 2, n_classes)

        self.c5 = nn.ConvTranspose2d(ch * 2, ch    , kernel_size=4, stride=2, padding=1, bias=False)
        self.i5 = CondInstanceNorm2d(ch, n_classes)

        attns = []
        for i in range(4):
            attns.append(nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid()))
        self.attns = nn.ModuleList(attns)

        self.c6 = nn.Conv2d(ch, 3, kernel_size=7, stride=1, padding=3, bias=False)


    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        h = x
        h = self.c1(h)
        h = self.activation(self.i1(h, c))

        h = self.c2(h)
        h = self.activation(self.i2(h, c))

        h = self.c3(h)
        h = self.activation(self.i3(h, c))

        for i in range(self.n_repeat):
            h = self.blocks[i](h, c)

        h = self.c4(h)
        h = self.activation(self.i4(h, c))

        h = self.c5(h)
        h = self.activation(self.i5(h, c))

        attns = []
        for i in range(4):
            attns.append(self.attns[i](h))

        h = self.multiViewSample(x, c, attns) + h

        out = F.tanh(self.c6(h))

        return out