import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)


class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool1d = nn.MaxPool1d(2)
    def forward(self, x):
        # c, h, w = x.size()
        x = self.forward_block(x)
        if self.pooling:
            x = self.pool1d(x)
        # _, c, h, w = x.size()
        return x
