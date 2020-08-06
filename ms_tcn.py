# This file directly follows MS-TCN https://github.com/yabufarha/ms-tcn
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStage(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStage(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out,_ = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            #out,_ = s(torch.cat((F.softmax(out, dim=1) * mask[:, 0:1, :],x),dim=1), mask)  # + video feature
            out,_ = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStage(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStage, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, mask)
        out = self.conv_out(feature) * mask[:, 0:1, :]
        return out, feature


class DilatedLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()  # default value is 0.5
        self.bn=nn.BatchNorm1d(in_channels, eps=1e-08, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x, mask,use_bn=False):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        if use_bn:
            out=self.bn(out)  # bn can not produce better result because of small batch size
        else:
            out = self.dropout(out)  # MS-TCN uses dropout, and we will follow their setting
        return (x + out) * mask[:, 0:1, :]  # residual, mask used for batch size >1