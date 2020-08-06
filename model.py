import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from lbp import LocalBarrierPooling
from bgm_block.models import fullBGM

class CascadeModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, dataset, device, use_lbp,num_soft_lbp):
        super(CascadeModel, self).__init__()
        self.num_stages= num_stages # number of cascade stages
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes) # cascade stage 1
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, dim + (s+1) * num_f_maps, num_classes)) for s in range(num_stages-1)]) # cascade stage 2,...,n
        self.stageF = SingleStageModel(num_layers, 64, num_classes, num_classes) # fusion stage
        self.bgm=fullBGM()
        self.lbp_in = LocalBarrierPooling(7, alpha=1)
        self.use_lbp=use_lbp
        self.num_soft_lbp=num_soft_lbp
        self.device = device
        if dataset=='50salads':
            self.lbp_out = LocalBarrierPooling(99, alpha=0.2) # has lbp_post
        if dataset=='breakfast':
            self.lbp_out = LocalBarrierPooling(159, alpha=0.3) # has lbp_post
        if dataset=='gtea':
            self.lbp_out = LocalBarrierPooling(99, alpha=1) # no lbp_post for gtea (because of bad barrier quality of resized BGM due to small dataset size), so alpha=1

    def forward(self, x, mask, gt_target=None, soft_threshold=0.8):
        mask.require_grad=False
        x.require_grad = False
        adjusted_weight=mask[:, 0:1, :].clone().detach().unsqueeze(0) # weights for SC
        for i in range(self.num_stages-1):
            adjusted_weight=torch.cat((adjusted_weight,mask[:, 0:1, :].clone().detach().unsqueeze(0)))
        #print(adjusted_weight.size())
        confidence=[]
        feature=[]
        if gt_target is not None:
            gt_target = gt_target.unsqueeze(0)

        # stage 1
        out1,feature1 = self.stage1(x, mask)
        outputs = out1.unsqueeze(0)
        feature.append(feature1)
        confidence.append(F.softmax(out1, dim=1)* mask[:, 0:1, :])
        confidence[0].require_grad=False

        if gt_target is None:
            max_conf,_ = torch.max(confidence[0],dim=1)
            max_conf = max_conf.unsqueeze(1).clone().detach()
            max_conf.require_grad=False
            decrease_flag=(max_conf>soft_threshold).float()
            increase_flag=mask[:, 0:1, :].clone().detach() - decrease_flag
            adjusted_weight[1]=max_conf.neg().exp() * decrease_flag + max_conf.exp() *increase_flag # for stage 2
        else:
            gt_conf=torch.gather(confidence[0],dim=1,index=gt_target)
            decrease_flag = (gt_conf > soft_threshold).float()
            increase_flag = mask[:, 0:1, :].clone().detach() - decrease_flag
            adjusted_weight[1]=gt_conf.neg().exp() * decrease_flag + gt_conf.exp() *increase_flag

        # stage 2,...,n
        curr_stage = 0
        for s in self.stages:
            curr_stage = curr_stage+ 1
            temp = feature[0]
            for i in range(1,len(feature)):
                temp= torch.cat((temp,feature[i]), dim=1) * mask[:, 0:1, :]
            temp = torch.cat((temp, x), dim=1)
            curr_out, curr_feature = s(temp, mask)
            outputs = torch.cat((outputs, curr_out.unsqueeze(0)), dim=0)
            feature.append(curr_feature)
            confidence.append(F.softmax(curr_out, dim=1) * mask[:, 0:1, :])
            confidence[curr_stage].require_grad = False
            if curr_stage==self.num_stages-1: # curr_stage starts from 0
                break  # don't need to compute the next stage's confidence when current stage = last cascade stage

            if gt_target is None:
                max_conf, _ = torch.max(confidence[curr_stage], dim=1)
                max_conf = max_conf.unsqueeze(1).clone().detach()
                max_conf.require_grad = False
                decrease_flag = (max_conf > soft_threshold).float()
                increase_flag = mask[:, 0:1, :].clone().detach() - decrease_flag
                adjusted_weight[curr_stage+1] = max_conf.neg().exp() * decrease_flag + max_conf.exp() * increase_flag # output the weight for the next stage
            else:
                gt_conf = torch.gather(confidence[curr_stage], dim=1, index=gt_target)
                decrease_flag = (gt_conf > soft_threshold).float()
                increase_flag = mask[:, 0:1, :].clone().detach() - decrease_flag
                adjusted_weight[curr_stage+1] = gt_conf.neg().exp() * decrease_flag + gt_conf.exp() * increase_flag

        output_weight=adjusted_weight.detach()
        output_weight.require_grad=False
        adjusted_weight = adjusted_weight / torch.sum(adjusted_weight, 0) # normalization among stages
        temp = F.softmax(out1, dim=1) * adjusted_weight[0]
        for i in range(1, self.num_stages):
            temp += F.softmax(outputs[i], dim=1) * adjusted_weight[i]
        confidenceF = temp * mask[:, 0:1, :] # input of fusion stage

        #  Inner LBP for confidenceF
        barrier, BGM_output = self.fullBarrier(x)
        if self.use_lbp:
            confidenceF = self.lbp_in(confidenceF, barrier)

        #  fusion stage: for more consistent output because of the combination of cascade stages may have much fluctuations
        out, _ = self.stageF(confidenceF, mask)    # use mixture of cascade stages

        #  Final LBP for output
        if self.use_lbp:
            for i in range(self.num_soft_lbp):
                out=self.lbp_out(out,barrier)

        confidence_last = torch.clamp(F.softmax(out, dim=1),min=1e-4 ,max=1- 1e-4) * mask[:, 0:1, :] # torch.clamp for training stability
        outputs = torch.cat((outputs, confidence_last.unsqueeze(0)), dim=0)
        return outputs,BGM_output,output_weight

    def fullBarrier(self,feature_tensor):
        BGM_output = self.bgm(feature_tensor)
        barrier = BGM_output
        return barrier,BGM_output

class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, mask)
        out = self.conv_out(feature) * mask[:, 0:1, :]
        return out, feature * mask[:, 0:1, :]


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()  # default value is 0.5
        self.bn=nn.BatchNorm1d(in_channels, eps=1e-08, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x, mask,use_bn=False):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        if use_bn:
            out=self.bn(out)
        else:
            out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]
