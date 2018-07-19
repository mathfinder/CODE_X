import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from source.dis_model.resblock import Block, OptimizedBlock, SNBlock, SNOptimizedBlock
from source.modules.sn_linear import SNLinear
from source.modules.sn_conv2d import SNConv2d
from source.modules.sn_embedding import SNEmbedding
import numpy as np

class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)
        self.conv3 = nn.Conv2d(curr_dim, 1, kernel_size=k_size, bias=False)

    def forward(self, x, c=None):
        h = self.main(x)
        out_real = self.conv1(h).squeeze(2).squeeze(2)
        out_aux = self.conv2(h).squeeze(2).squeeze(2)
        out_reg = F.tanh(self.conv3(h)).squeeze(2).squeeze(2) * 2.
        return out_real, out_aux, out_reg

class ProjectionDiscriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(ProjectionDiscriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.l1 = nn.Linear(curr_dim, 1)
        if c_dim > 0:
            self.l_y = nn.Embedding(c_dim, curr_dim)

        # self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)
        # self.conv3 = nn.Conv2d(curr_dim, 1, kernel_size=k_size, bias=False)
        self.l2 = nn.Linear(curr_dim, c_dim)
        self.l3 = nn.Linear(curr_dim, 1)

    def forward(self, x, c=None):
        h = self.main(x)
        h_mean = torch.mean(torch.mean(h, 3), 2)
        w_y = self.l_y(c)
        w_y = w_y / torch.norm(w_y, 2, dim=1, keepdim=True)

        out_real = self.l1(h_mean) + torch.sum(w_y * h_mean, dim=1, keepdim=True)
        # out_aux = self.conv2(h).squeeze(2).squeeze(2)
        # out_reg = F.tanh(self.conv3(h)).squeeze(2).squeeze(2) * 2.
        out_aux = self.l2(h_mean)
        out_reg = self.l3(h_mean)
        return out_real, out_aux, out_reg

class VariableDiscriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=3, n_hiddens=8):
        super(VariableDiscriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        n_dim = conv_dim * (2 ** (repeat_num-1))
        self.fc_mu = nn.Linear(n_dim, n_hiddens)
        self.fc_var = nn.Linear(n_dim, n_hiddens)
        self.fc1 = nn.Sequential(*[nn.Linear(n_hiddens*2 + 1, n_dim),
                                   nn.LeakyReLU(0.01, inplace=True)])
        self.fc2_1 = nn.Sequential(*[nn.Linear(n_dim, n_dim // 2),
                                     nn.LeakyReLU(0.01, inplace=True)])
        self.fc2_2 = nn.Sequential(*[nn.Linear(n_dim, n_dim // 2),
                                     nn.LeakyReLU(0.01, inplace=True)])
        self.fcl = nn.Sequential(*[nn.Linear(n_dim, 1)])

        if c_dim > 0:
            self.l_y1 = nn.Embedding(c_dim, curr_dim // 2)
            self.l_y2 = nn.Embedding(c_dim, curr_dim // 2)

    def get_z_random(self, batchSize, nz, random_type='const'):
        z = torch.Tensor(batchSize, nz)
        if random_type == 'uni':
            z.uniform_(-1., 1.)
        elif random_type == 'gauss':
            z.normal_(0., 1.)
        elif random_type == 'const':
            z.fill_(0.)
        z = Variable(z)
        return z
    def single_forward(self, x):
        h = self.main(x)
        h_mean = torch.mean(torch.mean(h, 3), 2)

        mu, logvar = self.fc_mu(h_mean), self.fc_var(h_mean)
        std = logvar.mul(0.5).exp()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss').type_as(std)
        return eps.mul(std).add_(mu), mu, logvar

    def forward(self, x1, x2, c1=None, c2=None):

        h1, mu1, logvar1 = self.single_forward(x1)
        h2, mu2, logvar2 = self.single_forward(x2)
        # h = torch.cat([h1, h2, F.pairwise_distance(h1, h2, keepdim=True), F.cosine_similarity(h1, h2).unsqueeze(1)], dim=1)
        h = torch.cat([h1, h2, F.cosine_similarity(h1, h2).unsqueeze(1)], dim=1)
        h = self.fc1(h)
        h1 = self.fc2_1(h)
        h2 = self.fc2_2(h)
        if c1 is not None:
            w_y1 = self.l_y1(c1)
            w_y1 = w_y1 / torch.norm(w_y1, 2, dim=1, keepdim=True)
        else:
            w_y1 = 0
        if c2 is not None:
            w_y2 = self.l_y2(c2)
            w_y2 = w_y2 / torch.norm(w_y2, 2, dim=1, keepdim=True)
        else:
            w_y2 = 0

        out_real = self.fcl(torch.cat([h1, h2], dim=1)) + torch.sum(w_y1*h1 + w_y2*h2, dim=1, keepdim=True)

        return out_real, mu1, logvar1, mu2, logvar2

class PairwiseDiscriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=3, n_hiddens=8):
        super(PairwiseDiscriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.InstanceNorm2d(curr_dim))
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        n_dim = conv_dim * (2 ** (repeat_num-1))
        self.fc_ct = nn.Linear(n_dim, n_hiddens)
        self.fc1 = nn.Sequential(*[nn.Linear(n_hiddens*2 + 1, n_dim),
                                   nn.LeakyReLU(0.01, inplace=True)])
        self.fc2_1 = nn.Sequential(*[nn.Linear(n_dim, n_dim // 2)])
        self.fc2_2 = nn.Sequential(*[nn.Linear(n_dim, n_dim // 2)])
        self.fcl = nn.Sequential(*[nn.Linear(n_dim, 1)])

        if c_dim > 0:
            self.l_y1 = nn.Sequential(*[nn.Embedding(c_dim, curr_dim // 2)])
            self.l_y2 = nn.Sequential(*[nn.Embedding(c_dim, curr_dim // 2)])

    def get_z_random(self, batchSize, nz, random_type='const'):
        z = torch.Tensor(batchSize, nz)
        if random_type == 'uni':
            z.uniform_(-1., 1.)
        elif random_type == 'gauss':
            z.normal_(0., 0.02)
        elif random_type == 'const':
            z.fill_(0.)
        z = Variable(z)
        return z
    def single_forward(self, x):
        h = self.main(x)
        h_mean = torch.mean(torch.mean(h, 3), 2)

        ct = self.fc_ct(h_mean)
        # ct = ct / (torch.norm(ct, 2, dim=1, keepdim=True) + e)

        return ct

    def forward(self, x1, x2, c1=None, c2=None, e=1e-5):

        h1 = self.single_forward(x1)
        h2 = self.single_forward(x2)
        h = torch.cat([h1, h2, F.cosine_similarity(h1, h2).unsqueeze(1)], dim=1)
        h = self.fc1(h)
        h1 = self.fc2_1(h)
        h2 = self.fc2_2(h)

        w_y1 = self.l_y1(c1)
        w_y1 = w_y1 / (torch.norm(w_y1, 2, dim=1, keepdim=True)+e)

        w_y2 = self.l_y2(c2)
        w_y2 = w_y2 / (torch.norm(w_y2, 2, dim=1, keepdim=True)+e)

        out_real = self.fcl(torch.cat([h1, h2], dim=1)) + torch.sum(w_y1*h1 + w_y2*h2, dim=1, keepdim=True)*0.5
        # debug:
        # print('self.fcl:', self.fcl(torch.cat([h1, h2], dim=1)))
        # print('w_y1*h1:', torch.sum(w_y1*h1, dim=1, keepdim=True))
        # print('w_y2*h2:', torch.sum(w_y2*h2, dim=1, keepdim=True))

        return out_real