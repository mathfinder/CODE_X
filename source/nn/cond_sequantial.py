import torch
import torch.nn as nn

class CondSequential(nn.Sequential):
    def __init__(self, *args):
        super(CondSequential, self).__init__(*args)

    def forward(self, input, cond):
        for module in self._modules.values():
            input = module(input, cond)
        return input
