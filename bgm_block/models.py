# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import copy

class fullBGM(torch.nn.Module):
    def __init__(self):
        super(fullBGM, self).__init__()
        self.feat_dim = 2048
        self.batch_size = 1
        self.c_hidden = 256
        self.bgm_best_loss = 10000000
        self.bgm_best_f1 = -10000000
        self.bgm_best_precision = -10000000
        self.output_dim = 1
        self.num_layers=3
        self.conv_in = nn.Conv1d(self.feat_dim, self.c_hidden, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** (i+2), self.c_hidden, self.c_hidden)) for i in range(self.num_layers)])
        self.conv_out = nn.Conv1d(self.c_hidden, self.output_dim, 1)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        out = torch.sigmoid(0.01*out)
        return out

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()  # default value is 0.5

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)

class resizedBGM(torch.nn.Module):
    def __init__(self, dataset):
        super(resizedBGM, self).__init__()
        self.feat_dim = 2048
        if dataset == 'breakfast' or dataset == 'gtea':
            self.temporal_dim = 300
        elif dataset == '50salads':
            self.temporal_dim = 400
        self.batch_size = 40
        self.batch_size_test = 10
        self.c_hidden = 512
        self.bgm_best_loss = 10000000
        self.bgm_best_f1= -10000000
        self.output_dim = 1
        self.conv1 = torch.nn.Conv1d(in_channels=self.feat_dim, out_channels=self.c_hidden, kernel_size=3, stride=1,
                                     padding=1, groups=1)
        self.conv2 = torch.nn.Conv1d(in_channels=self.c_hidden, out_channels=self.c_hidden, kernel_size=3, stride=1,
                                     padding=1, groups=1)
        self.conv3 = torch.nn.Conv1d(in_channels=self.c_hidden, out_channels=self.output_dim, kernel_size=1, stride=1,
                                     padding=0)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x


