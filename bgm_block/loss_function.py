# -*- coding: utf-8 -*-
import torch

def bi_loss(scores, anchors, bgm_match_threshold=0.5):
    '''
    cross_entropy loss
    :param scores: gt
    :param anchors: predict result
    :param bgm_match_threshold: threshold for selecting positive samples
    :return:
    '''
    scores = scores.view(-1).cuda()
    anchors = anchors.contiguous().view(-1)
    pmask = (scores> bgm_match_threshold).float().cuda()
    num_positive = torch.sum(pmask)
    num_entries = len(scores)
    ratio=num_entries/num_positive

    coef_0=0.5*(ratio)/(ratio-1)
    coef_1=coef_0*(ratio-1)
    loss = coef_1*pmask*torch.log(anchors+0.00001) + coef_0*(1.0-pmask)*torch.log(1.0-anchors*0.999999)
    loss=-torch.mean(loss)
    num_sample=[torch.sum(pmask),ratio] 
    return loss,num_sample

def BGM_loss_calc(anchors,match_scores):
    loss_start_small,num_sample_start_small=bi_loss(match_scores,anchors)
    loss_dict={"loss":loss_start_small,"num_sample":num_sample_start_small}
    return loss_dict

def BGM_loss_function(y,BGM_output):
    loss_dict = BGM_loss_calc(BGM_output, y)
    cost=loss_dict["loss"]
    loss_dict["cost"] = cost
    return loss_dict

def BGM_cal_P_R(y,BGM_output):
    precision,recall = cal_P_R(BGM_output,y)
    return precision,recall

def cal_P_R(anchors, scores, acc_threshold=0.5):
    scores = scores.view(-1)
    anchors = anchors.contiguous().view(-1)
    output = (anchors > acc_threshold).int().cpu()
    gt=(scores > acc_threshold).int().cpu()
    TP=0.0
    FP=0.0
    FN=0.0
    if scores.size()[0]==0:
        return 0.0,0.0
    for i in range(scores.size()[0]):
        if output[i]==1:
            if output[i]==gt[i]:
                TP=TP+1
            else:
                FP=FP+1
        else:
            if gt[i]==1:
                FN=FN+1
    if (TP+FP)==0:
        return 0.0,0.0
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    return precision,recall

