import torch
import torch.nn as nn
import torch.nn.functional as F
from source.gen_model.resblock import OptimizedBlock, Block
from source.nn.self_attention import SelfAttention
from source.nn.multi_view import MultiView
from source.nn.multi_sample import MultiSample
from source.nn.condinstancenorm import CondInstanceNorm2d


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, mode='add'):
        super(ResidualBlock, self).__init__()
        self.mode = mode
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        if self.mode == 'add':
            return x + self.main(x)
        elif self.mode == 'multi':
            return x * self.main(x) + x

class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class MultiViewGenerator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(MultiViewGenerator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(MultiView(ch_in=curr_dim, ch_out=3, if_attention=True))
        # layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        self.main = nn.Sequential(*layers)
        layers.append(nn.Tanh())


    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        return self.main(x)

class MultiSampleGenerator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(MultiSampleGenerator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))

        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=1, bias=False))
        layers.append(MultiView(ch_in=conv_dim, ch_out=3, if_attention=False))
        self.aux = nn.Sequential(*layers)


    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        deep = self.main(x)
        shallow = self.aux(x)
        return F.tanh((deep + shallow) / 2.)

class CrossAttentionMultiSampleGenerator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(CrossAttentionMultiSampleGenerator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)
        self.deep = nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        attns = []
        for i in range(5):
            attns.append(nn.Sequential(nn.Conv2d(curr_dim, 1, 1), nn.Sigmoid()))
        self.attns = nn.ModuleList(attns)

        self.sample1 = nn.Conv2d(3 + c_dim, conv_dim, kernel_size=1, bias=False)
        self.sample2 = MultiSample(ch_in=conv_dim, ch_out=3)
        self.smooth = nn.Conv2d(3 * 6 + c_dim, 3, kernel_size=1, bias=False)


    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        mid = self.main(x)
        deep = self.deep(mid)
        attns = []
        for i in range(5):
            attns.append(self.attns[i](mid))

        smaple = self.sample1(x)
        sample = self.sample2(smaple, attention=attns)
        return F.tanh(self.smooth(torch.cat([deep, sample, c], dim=1)))

class CINGenerator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, ch=64, n_classes=5, n_repeat=6, activation=F.relu):
        super(CINGenerator, self).__init__()
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
        if c.dim() > 1 and c.size()[1]>1:
            c = c.argmax(dim=1)
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

class CINCrossAttentionMultiSampleGenerator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, ch=64, n_classes=5, n_repeat=6, activation=F.relu):
        super(CINCrossAttentionMultiSampleGenerator, self).__init__()
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

        attns = []
        for i in range(5):
            attns.append(nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid()))
        self.attns = nn.ModuleList(attns)

        self.sample1 = nn.Conv2d(3 + n_classes, ch, kernel_size=1, bias=False)
        self.sample2 = MultiSample(ch_in=ch, ch_out=3)
        self.smooth = nn.Conv2d(3 * 6 + n_classes, 3, kernel_size=1, bias=False)


    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c_idx = c.argmax(dim=1)
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x_c = torch.cat([x, c], dim=1)

        h = x
        h = self.c1(h)
        h = self.activation(self.i1(h, c_idx))

        h = self.c2(h)
        h = self.activation(self.i2(h, c_idx))

        h = self.c3(h)
        h = self.activation(self.i3(h, c_idx))

        for i in range(self.n_repeat):
            h = self.blocks[i](h, c_idx)

        h = self.c4(h)
        h = self.activation(self.i4(h, c_idx))

        h = self.c5(h)
        h = self.activation(self.i5(h, c_idx))

        deep = self.c6(h)
        attns = []
        for i in range(5):
            attns.append(self.attns[i](h))

        smaple = self.sample1(x_c)
        sample = self.sample2(smaple, attention=attns)

        return F.tanh(self.smooth(torch.cat([deep, sample, c], dim=1)))

class CINSelfCrossAttentionMultiSampleGenerator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, ch=64, n_classes=5, n_repeat=6, activation=F.relu):
        super(CINSelfCrossAttentionMultiSampleGenerator, self).__init__()
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

        self.SelfAttns = SelfAttention(ch * 4, ch // 2)

        # Up-Sampling
        self.c4 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.i4 = CondInstanceNorm2d(ch * 2, n_classes)

        self.c5 = nn.ConvTranspose2d(ch * 2, ch    , kernel_size=4, stride=2, padding=1, bias=False)
        self.i5 = CondInstanceNorm2d(ch, n_classes)

        self.c6 = nn.Conv2d(ch, 3, kernel_size=7, stride=1, padding=3, bias=False)

        attns = []
        for i in range(5):
            attns.append(nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid()))
        self.attns = nn.ModuleList(attns)

        self.sample1 = nn.Conv2d(3 + n_classes, ch, kernel_size=1, bias=False)
        self.sample2 = MultiSample(ch_in=ch, ch_out=3)
        self.smooth = nn.Conv2d(3 * 6 + n_classes, 3, kernel_size=1, bias=False)


    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c_idx = c.argmax(dim=1)
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x_c = torch.cat([x, c], dim=1)

        h = x
        h = self.c1(h)
        h = self.activation(self.i1(h, c_idx))

        h = self.c2(h)
        h = self.activation(self.i2(h, c_idx))

        h = self.c3(h)
        h = self.activation(self.i3(h, c_idx))

        for i in range(self.n_repeat):
            h = self.blocks[i](h, c_idx)
        h = self.SelfAttns(h) + h
        h = self.c4(h)
        h = self.activation(self.i4(h, c_idx))

        h = self.c5(h)
        h = self.activation(self.i5(h, c_idx))

        deep = self.c6(h)
        attns = []
        for i in range(5):
            attns.append(self.attns[i](h))

        smaple = self.sample1(x_c)
        sample = self.sample2(smaple, attention=attns)

        return F.tanh(self.smooth(torch.cat([deep, sample, c], dim=1)))

class CINCrossAttentionMultiSampleFullResidualGenerator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, ch=64, n_classes=5, n_repeat=6, activation=F.relu):
        super(CINCrossAttentionMultiSampleFullResidualGenerator, self).__init__()
        self.n_repeat = n_repeat
        self.activation=activation
        self.c1 = nn.Conv2d(3, ch, kernel_size=7, stride=1, padding=3, bias=False)

        # Down-Sampling
        self.block1 = Block(ch, ch * 2, activation=activation, downsample=True, n_classes=n_classes)
        self.block2 = Block(ch * 2, ch * 4, activation=activation, downsample=True, n_classes=n_classes)

        # Bottleneck
        blocks = []
        for i in range(n_repeat):
            blocks.append(Block(ch * 4, ch * 4, activation=activation, n_classes=n_classes))
        self.blocks = nn.ModuleList(blocks)

        # Up-Sampling
        self.block3 = Block(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
        self.block4 = Block(ch * 2, ch    , activation=activation, upsample=True, n_classes=n_classes)

        self.c2 = nn.Conv2d(ch, 3, kernel_size=7, stride=1, padding=3, bias=False)

        attns = []
        for i in range(5):
            attns.append(nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid()))
        self.attns = nn.ModuleList(attns)

        self.sample1 = nn.Conv2d(3 + n_classes, ch, kernel_size=1, bias=False)
        self.sample2 = MultiSample(ch_in=ch, ch_out=3)
        self.smooth = nn.Conv2d(3 * 6 + n_classes, 3, kernel_size=1, bias=False)


    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c_idx = c.argmax(dim=1)
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x_c = torch.cat([x, c], dim=1)

        h = x
        h = self.c1(h)

        h = self.block1(h, c_idx)
        h = self.block2(h, c_idx)

        for i in range(self.n_repeat):
            h = self.blocks[i](h, c_idx)

        h = self.block3(h, c_idx)
        h = self.block4(h, c_idx)

        deep = self.c2(h)
        attns = []
        for i in range(5):
            attns.append(self.attns[i](h))

        smaple = self.sample1(x_c)
        sample = self.sample2(smaple, attention=attns)

        return F.tanh(self.smooth(torch.cat([deep, sample, c], dim=1)))