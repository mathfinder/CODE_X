import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

def loss_dcgan_dis(dis_fake, dis_real):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    loss = L1 + L2
    return loss

def loss_dcgan_gen(dis_fake):
    loss = torch.mean(F.softplus(-dis_fake))
    return loss

def loss_hinge_dis(dis_fake, dis_real):
    loss = torch.mean(F.relu(1. - dis_real))
    loss += torch.mean(F.relu(1. + dis_fake))
    return loss

def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss

def loss_hard_reg(pred, label, n_classes):
    n_intervals = n_classes - 1
    target = label.reshape(label.size()[0], 1).float() / (n_intervals * 0.5) - 1.
    # loss = F.mse_loss(pred, target)
    loss = torch.mean(torch.abs(pred-target))
    return loss

def loss_soft_reg(pred, label):
    tmp = pred.squeeze()
    loss = torch.mean( F.relu(tmp - label.float() - 1.) + F.relu(label.float() - tmp))
    return loss

def loss_kl(mu, logvar):
    loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # loss = torch.mean(mu.pow(2))
    return loss


