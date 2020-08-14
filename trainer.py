# Adapted from MS-TCN, function train_mstcn and predict_mstcn directly use their implementation in
# https://github.com/yabufarha/ms-tcn
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tensorboardX import SummaryWriter
from torch.distributions import Categorical
from model import CascadeModel
from ms_tcn import MultiStageModel
from torch import optim
import numpy as np
from lbp import LocalBarrierPooling
from eval import eval_metric
from bgm_block.loss_function import BGM_loss_function,BGM_cal_P_R


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes,dataset,device,use_lbp,num_soft_lbp):
        self.num_stages=num_blocks-1  # num_blocks = number of cascade stages + 1 fusion stage
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes) # ms-tcn model
        self.cascadeModel = CascadeModel(self.num_stages,num_layers, num_f_maps, dim, num_classes,dataset,device,use_lbp,num_soft_lbp) # our bcn model
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)  # for ms_tcn
        self.maskCE = nn.CrossEntropyLoss(ignore_index=-100,reduction='none')  # for cascade stages
        self.nll=nn.NLLLoss(ignore_index=-100,reduction='none')   # for fusion stage
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train_bcn(self, save_dir, data_loader,  num_epochs, learning_rate, device, date, num,checkpoint_path, results_dir, features_path,actions_dict, sample_rate, dataset,
                  vid_list_file_tst, ground_truth_path, split, use_lbp, bgm_result_path, pooling_length, num_post=4):

        writer = SummaryWriter('tensorboardX/run%s_%s_%s' % (date, num, split))
        self.cascadeModel.train()
        self.cascadeModel.bgm.load_state_dict(torch.load(checkpoint_path + "/bgm_best_f1.model"))
        self.cascadeModel.to(device)

        # different learning rate for SC and BGM
        optimizer = optim.Adam([{'params': self.cascadeModel.stage1.parameters()},
                                {'params': self.cascadeModel.stages.parameters()},
                                {'params': self.cascadeModel.stageF.parameters()},
                                {'params': self.cascadeModel.bgm.parameters(), 'lr': learning_rate*0.1}
                                ], lr=learning_rate)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.3)
        if dataset=="gtea":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.3)

        lbp = LocalBarrierPooling(pooling_length, alpha=2)
        lbp = lbp.to(device)
        # load features of all test videos in memory, change it if not enough CPU memory
        inverse_dict = {v: k for k, v in actions_dict.items()}
        file_ptr = open(vid_list_file_tst, 'r')
        list_of_vids_tst = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        all_test_feature = []
        for vid in list_of_vids_tst:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = torch.Tensor(features[:, :: sample_rate]).unsqueeze_(0).cpu()
            all_test_feature.append(features)
        print("Testing set result format: Acc, Edit, F1@10, F1@25, F1@50")
        for epoch in range(num_epochs):
            self.cascadeModel.train()
            scheduler.step()
            epoch_loss = 0 # loss printed in screen
            # calculate metrics
            correct = 0
            total = 0
            precision = 0
            recall = 0
            # freeze the parameters in the first several epochs
            freeze_epochs = 15
            if dataset=='breakfast':
                freeze_epochs=20
            elif dataset=='gtea':
                freeze_epochs = 18
            if epoch<freeze_epochs:
                for param in self.cascadeModel.bgm.parameters():
                    param.requires_grad_(False)
            else:
                for param in self.cascadeModel.bgm.parameters():
                    param.requires_grad_(True)
            for n_iter, (batch_input, batch_target, mask, label) in enumerate(data_loader):
                batch_input, batch_target, mask = batch_input.to(device),batch_target.to(device), mask.to(device)
                predictions,BGM_output,adjust_weight = self.cascadeModel(batch_input, mask, gt_target=batch_target,soft_threshold=0.8)

                loss = 0
                balance_weight = [1.0]*self.num_stages
                # num_stages is number of cascade stages
                for num_stage in range(self.num_stages):
                    adjust_weight[num_stage].require_grad=False
                    balance_weight[num_stage] = torch.mean(torch.sum(adjust_weight[0],2).view(-1).float() / torch.sum(adjust_weight[num_stage],2).view(-1).float())
                    p = predictions[num_stage]
                    loss += 1 * balance_weight[num_stage] * torch.mean(adjust_weight[num_stage].view(-1) * self.maskCE(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1)))
                    loss += 0.3 * torch.mean(
                        torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=8) * mask[:, :, 1:])

                # fusion stage
                p = predictions[-1]
                loss += torch.mean(self.nll(torch.log(p.transpose(2, 1).contiguous().view(-1, self.num_classes)), batch_target.view(-1)))
                loss += 0.5 * torch.mean(torch.clamp(
                    self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                    max=8) * mask[:, :, 1:])

                # calculate bgm's performance, but we don't use bgm's loss here and update bgm's parameters with backward gradients
                #loss += BGM_loss_function(label, BGM_output)["cost"]   # uncomment this line to add bgm's loss, but we think it is unnecessary
                batch_precision, batch_recall = BGM_cal_P_R(label, BGM_output)
                precision = precision + batch_precision
                recall = recall + batch_recall

                optimizer.zero_grad()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()


            # statistics about training data
            print("[epoch %d]: training loss = %.3f, training set acc=%.3f ,lr=%.4f" % (
                epoch + 1, epoch_loss / len(data_loader), float(correct) / total, optimizer.param_groups[0]['lr']))
            num_iter=len(data_loader)-1

            if (precision + recall) > 0:
                f1_score = 2 * (precision / (num_iter + 1)) * (recall / (num_iter + 1)) / ((precision / (num_iter + 1)) + (recall / (num_iter + 1)))
                print("BGM in training set: P=%.03f, R=%.03f, f1=%.3f" % (precision / (num_iter + 1), recall / (num_iter + 1), f1_score))
            else:
                print("BGM in training set: P=%.03f, R=%.03f" % ( precision / (num_iter + 1), recall / (num_iter + 1)))
            writer.add_scalar('Train_loss', epoch_loss / len(data_loader), epoch)
            writer.add_scalar('Train_Acc', float(correct) / total, epoch)
            writer.add_scalar('Train_BGM_Precision', precision / (num_iter + 1), epoch)
            writer.add_scalar('Train_BGM_Recall', recall / (num_iter + 1), epoch)

            # statistics about testing data, used for selecting epochs
            self.cascadeModel.eval()
            for i in range(len(list_of_vids_tst)):
                vid = list_of_vids_tst[i]
                input_x = all_test_feature[i]
                input_x = input_x.to(device)
                mask = torch.ones(input_x.size(), device=device)
                predictions, _, _ = self.cascadeModel(input_x, mask, gt_target=None, soft_threshold=0.8)
                predictions=predictions[-1]
                if use_lbp and dataset != "gtea":
                    num_frames = np.shape(input_x)[2]
                    barrier_file = bgm_result_path + vid + ".csv"
                    barrier = pd.read_csv(barrier_file)
                    barrier = np.transpose(np.array(barrier))
                    temporal_scale = np.shape(barrier)[1]
                    barrier = torch.Tensor(barrier)  # size=[1, num_frames]
                    if temporal_scale < num_frames:
                        interpolation = torch.round(torch.Tensor([float(num_frames) / temporal_scale * (i + 0.5) for i in range(temporal_scale)])).long()
                        resize_barrier = torch.Tensor([0.0] * num_frames)
                        resize_barrier[interpolation] = barrier[0]
                        resize_barrier = resize_barrier.unsqueeze(0).unsqueeze(0)  # size=[1,1,num_frames]
                    else:
                        resize_barrier = barrier
                        resize_barrier = resize_barrier.unsqueeze(0)  # size=[1,1,num_frames]
                    resize_barrier = resize_barrier.to(device)
                    if temporal_scale<num_frames:
                        for i in range(num_post):
                            predictions = lbp(predictions, resize_barrier)
                    else:
                        predictions=F.interpolate(predictions,size=temporal_scale, mode='linear',align_corners=False)
                        for i in range(num_post):
                            predictions = lbp(predictions, resize_barrier)
                        predictions = F.interpolate(predictions, size=num_frames, mode='linear',align_corners=False)

                _, predicted = torch.max(predictions.data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for k in range(len(predicted)):
                    recognition = np.concatenate((recognition, [inverse_dict[predicted[k].item()]] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

            test_acc, test_edit, test_f1 = eval_metric(dataset, list_of_vids_tst, ground_truth_path, results_dir + "/")
            writer.add_scalar('Test_Acc', test_acc, epoch)
            writer.add_scalar('Test_edit', test_edit, epoch)
            writer.add_scalar('Test_f1@10', test_f1[0], epoch)
            writer.add_scalar('Test_f1@25', test_f1[1], epoch)
            writer.add_scalar('Test_f1@50', test_f1[2], epoch)

            # save model
            if epoch >= 1 * num_epochs / 2 - 1 or epoch >= 25:
                torch.save(self.cascadeModel.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")

    def predict_bcn(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict,bgm_result_path, device, sample_rate, dataset, ground_truth_path,
                    pooling_length, use_lbp, num_post=4):
        self.cascadeModel.eval()
        inverse_dict={v: k for k, v in actions_dict.items()}
        lbp = LocalBarrierPooling(pooling_length, alpha=2)
        lbp = lbp.to(device)
        with torch.no_grad():
            self.cascadeModel.to(device)
            self.cascadeModel.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"), strict=True)
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                print (vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                if use_lbp and dataset != "gtea":
                    num_frames = np.shape(features)[1]
                    barrier_file=bgm_result_path+ vid + ".csv"
                    barrier=pd.read_csv(barrier_file)
                    #barrier = np.array([[0.0]*temporal_scale])  # avg_pooling
                    #barrier = np.array([[1.0]*temporal_scale])  # gaussian-like
                    barrier = np.transpose(np.array(barrier))
                    temporal_scale = np.shape(barrier)[1]
                    barrier = torch.Tensor(barrier)  # size=[1, num_frames]
                    if temporal_scale<num_frames:
                        interpolation = torch.round(torch.Tensor([float(num_frames) / temporal_scale * (i+0.5) for i in range(temporal_scale)])).long()
                        resize_barrier = torch.Tensor([0.0]*num_frames)
                        resize_barrier[interpolation]= barrier[0]
                        resize_barrier = resize_barrier.unsqueeze(0).unsqueeze(0)  # size=[1,1,num_frames]
                    else:
                        resize_barrier = barrier
                        resize_barrier = resize_barrier.unsqueeze(0)  # size=[1,1,num_frames]
                    resize_barrier=resize_barrier.to(device)

                input_x = torch.Tensor(features)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                mask = torch.ones(input_x.size(), device=device)

                predictions, BGM_output, _ = self.cascadeModel(input_x, mask, gt_target=None,soft_threshold=0.8)
                predictions = predictions[-1]
                if use_lbp and dataset != "gtea":
                    if temporal_scale<=num_frames:
                        for i in range(num_post):
                            predictions = lbp(predictions, resize_barrier)
                    else:
                        predictions=F.interpolate(predictions,size=temporal_scale, mode='linear',align_corners=False)
                        for i in range(num_post):
                            predictions = lbp(predictions, resize_barrier)
                        predictions = F.interpolate(predictions, size=num_frames, mode='linear',align_corners=False)

                # generate segmentation result
                _, predicted = torch.max(predictions.data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [inverse_dict[predicted[i].item()]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
            # calculate metrics and show in screen
            eval_metric(dataset,list_of_vids,ground_truth_path,results_dir+"/")

    def train_mstcn(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device, date,num):
        writer = SummaryWriter('run%s_%s' % (date,num))
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            #scheduler.step()
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)
                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            #torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f,    lr=%f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),float(correct)/total,optimizer.param_groups[0]['lr']))
            writer.add_scalar('Train_loss', epoch_loss / len(batch_gen.list_of_examples), epoch)
            writer.add_scalar('Acc', float(correct)/total, epoch)


    def predict_mstcn(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate,bsn_result_path,mstcn_use_lbp,poolingLength=99):
        self.model.eval()
        inverse_dict={v: k for k, v in actions_dict.items()}
        lbp = LocalBarrierPooling(poolingLength)
        lbp = lbp.to(device)

        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                print (vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                if mstcn_use_lbp:
                    num_frames = np.shape(features)[1]
                    barrier_file=bsn_result_path+vid+ ".csv"
                    barrier=np.array(pd.read_csv(barrier_file))
                    temporal_scale=np.shape(barrier)[0]
                    barrier=np.transpose(barrier)
                    barrier = torch.tensor(barrier, dtype=torch.float)  #size=[num_frames]
                    if temporal_scale<=num_frames:
                        resize_barrier = F.interpolate(barrier, size=num_frames, mode='nearest')
                    else:
                        resize_barrier=barrier
                    resize_barrier = resize_barrier.unsqueeze(0)
                    resize_barrier = resize_barrier.unsqueeze(0)  # size=[1,1,num_frames]
                    resize_barrier=resize_barrier.to(device)

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                predictions = predictions[-1]
                if mstcn_use_lbp:
                    if temporal_scale<=num_frames:
                        predictions=lbp(predictions,resize_barrier)
                    else:
                        predictions=F.interpolate(predictions,size=temporal_scale, mode='linear',align_corners=False)
                        predictions = lbp(predictions, resize_barrier)
                        predictions = F.interpolate(predictions, size=num_frames, mode='linear',align_corners=False)
                predictions=F.softmax(predictions,dim=1)
                entropy = Categorical(probs=predictions.squeeze(0).transpose(1, 0)).entropy()
                entropy = entropy.cpu().numpy().astype(np.str)

                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/entropy_" + f_name, "w")
                f_ptr.write(' '.join(entropy))
                f_ptr.close()

                _, predicted = torch.max(predictions.data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [inverse_dict[predicted[i].item()]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
