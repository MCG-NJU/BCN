import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def unfold_1d(x, kernel_size=7, pad_value=0):
    B, C, T = x.size()
    padding = kernel_size//2
    x = x.unsqueeze(-1)
    x = F.pad(x, (0, 0, padding, padding), value=pad_value)
    D = F.unfold(x, (kernel_size, 1), padding=(0, 0))
    return D.view(B, C, kernel_size, T)


def dual_barrier_weight(b, kernel_size=7,alpha=0.2):
    '''
    b: (B, 1, T)
    '''
    K = kernel_size
    b = unfold_1d(b, kernel_size=K, pad_value=20)
    # b: (B, 1, K, T)
    HL = K//2
    left = torch.flip(torch.cumsum(
        torch.flip(b[:, :, :HL+1, :], [2]), dim=2), [2])[:, :, :-1, :]
    right = torch.cumsum(b[:, :, -HL-1:, :], dim=2)[:, :, 1:, :]
    middle = torch.zeros_like(b[:, :, 0:1, :])
    #middle = b[:, :, HL:-HL, :]
    weight=alpha*torch.cat((left, middle, right), dim=2)
    return weight.neg().exp()


class LocalBarrierPooling(nn.Module):
    def __init__(self, kernel_size=99,alpha=0.2):
        super(LocalBarrierPooling, self).__init__()
        self.kernel_size = kernel_size
        self.alpha=alpha

    def forward(self, x, barrier):
        '''
        x: (B, C, T)
        barrier: (B, 1, T) (>=0)
        '''
        xs = unfold_1d(x, self.kernel_size)
        w = dual_barrier_weight(barrier, self.kernel_size, self.alpha)

        return (xs*w).sum(dim=2)/((w).sum(dim=2)+np.exp(-10))


if __name__ == "__main__":
    B = torch.Tensor([ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]).view(1, 1, -1)
    x = torch.Tensor([10,10,10,10,10,10,10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10]).view(1, 1, -1)
    lbp = LocalBarrierPooling(13,0.2)
    pooled_x = lbp(x, B)
    print(pooled_x)
