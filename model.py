import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from source.gen_model.resblock import OptimizedBlock, Block



class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, ch=64, n_classes=5, n_repeat=6, activation=F.relu):
        super(Generator, self).__init__()
        self.n_repeat = n_repeat

        self.block1 = OptimizedBlock(3 + n_classes, ch)
        self.block2 = Block(ch    , ch * 2, activation=activation, downsample=True)
        self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
        blocks = []
        for i in range(n_repeat):
            blocks.append(Block(ch * 4, ch * 4, activation=activation))
        self.blocks = nn.ModuleList(*blocks)
        self.block4 = Block(ch * 4, ch * 2, activation=activation, upsample=True)
        self.block5 = Block(ch * 2, ch    , activation=activation, upsample=True)

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
        out = x
        h = x
        h = self.block1(h)
        h = self.block2(h, c)
        h = self.block3(h, c)
        for i in range(self.repeat):
            h = self.blocks[i](h, c)
        h = self.block4(h, c)
        h = self.block5(h, c)

        out += self.views[0](h)
        for i in range(4):
            out += self.views[i+1](h) * self.attns[i](h).expand_as(x)

        return F.tanh(out)


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        return out_real.squeeze(2).squeeze(2), out_aux.squeeze(2).squeeze(2)

class PairDiscriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(PairDiscriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        return out_real.squeeze(2).squeeze(2), out_aux.squeeze(2).squeeze(2)