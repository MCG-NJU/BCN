# bgm.py and files in bgm_block are adapted from BSN's code: https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch
import sys
sys.dont_write_bytecode = True
import os
import torch
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from bgm_block.dataset import BoundaryDataset
from bgm_block.models import fullBGM, resizedBGM
from bgm_block.loss_function import BGM_loss_function,BGM_cal_P_R
import pandas as pd
import argparse
from scipy import signal
sys.path.append(os.path.curdir)

#os.environ["CUDA_VISIBLE_DEVICES"] = '4'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='test')
parser.add_argument('--dataset', default="breakfast")
parser.add_argument('--split', default='4')
parser.add_argument('--resolution', default='resized') # full or resized
parser.add_argument("--date", help="today", type=str, default='0524')
parser.add_argument("--num", help="how many times do you train today? It is only used for TensorBoardX", type=str,
                    default='1')

args = parser.parse_args()

print("%sing '%s' resolution, %s dataset, split %s"%(args.action,args.resolution,args.dataset,args.split))

use_saved_model=True

vid_list_file = "./data/" + args.dataset + "/splits/train.split" + args.split + ".bundle"
vid_list_file_tst = "./data/" + args.dataset + "/splits/test.split" + args.split + ".bundle"
features_path = "./data/" + args.dataset + "/features/"
gt_path = "./data/" + args.dataset + "/groundTruth/"
mapping_file = "./data/" + args.dataset + "/mapping.txt"
bgm_resized_result_path = "./bgm_result/resized/bgm_output/" + args.dataset + "/"
bgm_full_result_path = "./bgm_result/full/bgm_output/" + args.dataset + "/"
bgm_full_gt_path = "./bgm_result/full/bgm_gt/" + args.dataset + "/split_" + args.split + "/"
bgm_resized_gt_path = "./bgm_result/resized/bgm_gt/" + args.dataset + "/split_" + args.split + "/"
checkpoint_path = "./bgm_models/" + args.resolution + "/" + args.dataset + "/split_" + args.split

if use_saved_model:
    checkpoint_path = "./best_bgm_models/" + args.resolution + "/" + args.dataset + "/split_" + args.split

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)

sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.exists(bgm_full_result_path):
    os.makedirs(bgm_full_result_path)
if not os.path.exists(bgm_resized_result_path):
    os.makedirs(bgm_resized_result_path)
if not os.path.exists(bgm_full_gt_path):
    os.makedirs(bgm_full_gt_path)
if not os.path.exists(bgm_resized_gt_path):
    os.makedirs(bgm_resized_gt_path)

if args.resolution == "resized":
    use_full=False # True: full resolution; False: resized resolution
elif args.resolution == "full":
    use_full = True
else:
    print("invalid resolution for BGM!")

def main():
    if args.action == "train":
        BCN_Train_BGM(device,args.date,args.num,args.dataset)
    elif args.action == "test":
        if use_full:
            BCN_generate_barrier(device,args.dataset)
        else:
            BCN_generate_single_barrier(device,args.dataset)
    elif args.action == "gt":
        BCN_output_boundary_gt(use_full) # just for us to compare gt and test result by ourselves in csv file
    else:
        print("Invalid mode for BGM!")


def train_BGM(device,data_loader, model, optimizer, epoch, writer,dataset):
    model.train()
    num_iter =0
    epoch_cost = 0
    for n_iter, (input_data, label) in enumerate(data_loader):
        input_data=input_data.to(device)
        BGM_output = model(input_data)
        loss = BGM_loss_function(label,BGM_output)
        cost = loss["cost"]
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        epoch_cost += loss["cost"].cpu().detach().numpy()
        num_iter = n_iter
    writer.add_scalars('data/cost', {'train': epoch_cost / (num_iter + 1)}, epoch)
    if (epoch % 10 == 1) or dataset == 'breakfast':
        print("Train loss(epoch %d): %.03f,lr=%f" % (
        epoch, epoch_cost / (num_iter + 1),optimizer.param_groups[0]['lr']))


def test_BGM(device,data_loader, model, epoch, writer,dataset):
    model.eval()
    epoch_cost = 0
    precision=0
    recall = 0
    num_iter = 0
    for n_iter, (input_data, label) in enumerate(data_loader):
        input_data = input_data.to(device)
        BGM_output = model(input_data)
        loss = BGM_loss_function(label, BGM_output)

        batch_precision,batch_recall = BGM_cal_P_R(label, BGM_output)
        precision=precision+batch_precision
        recall = recall + batch_recall
        epoch_cost += loss["cost"].cpu().detach().numpy()
        num_iter = n_iter

    writer.add_scalars('data/precision', {'test': precision / (num_iter + 1)}, epoch)
    writer.add_scalars('data/recall', {'test': recall / (num_iter + 1)}, epoch)
    if (epoch % 10 == 1) or dataset == 'breakfast':
        if (precision + recall) > 0:
            f1_score = 2 * (precision / (num_iter + 1)) * (recall / (num_iter + 1)) / ((precision / (num_iter + 1)) + (recall / (num_iter + 1)))
            print("epoch %d: P=%.03f, R=%.03f, f1=%.3f" % (epoch, precision / (num_iter + 1), recall / (num_iter + 1), f1_score))
        else:
            print("epoch %d: P=%.03f, R=%.03f" % (epoch, precision / (num_iter + 1), recall / (num_iter + 1)))
    torch.save(model.state_dict(), checkpoint_path + "/bgm_last.model")
    if epoch_cost < model.bgm_best_loss:
        model.bgm_best_loss = np.mean(epoch_cost)
        torch.save(model.state_dict(), checkpoint_path + "/bgm_best_loss.model")

    if (precision + recall) > 0:
        f1_score = 2 * (precision / (num_iter + 1)) * (recall / (num_iter + 1)) / ((precision / (num_iter + 1)) + (recall / (num_iter + 1)))
        if f1_score > model.bgm_best_f1:
            model.bgm_best_f1 = f1_score
            torch.save(model.state_dict(), checkpoint_path + "/bgm_best_f1.model")

def BCN_Train_BGM(device, date, num, dataset, bsn_epochs = 300):
    learning_rate=0.001
    if dataset=='breakfast':
        learning_rate = 0.0001
        bsn_epochs=100
    if dataset=='gtea':
        learning_rate = 0.0002
        bsn_epochs=100
    writer = SummaryWriter('tensorboardX/run%s_%s' % (date,num))
    if use_full:
        model = fullBGM()
    else:
        model = resizedBGM(dataset)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    train_loader = torch.utils.data.DataLoader(BoundaryDataset("train",use_full,vid_list_file, num_classes, actions_dict, gt_path, features_path, sample_rate,args.dataset,device),
                                               batch_size=model.batch_size, shuffle=True,
                                               num_workers=4, pin_memory=True, drop_last=False)

    test_loader = torch.utils.data.DataLoader(BoundaryDataset("train",use_full,vid_list_file_tst, num_classes, actions_dict, gt_path, features_path, sample_rate,args.dataset,device),
                                              batch_size=1, shuffle=False,
                                              num_workers=4, pin_memory=True, drop_last=False)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(bsn_epochs / 3), int(2 * bsn_epochs / 3)],gamma=0.3)
    for epoch in range(bsn_epochs):
        scheduler.step()
        train_BGM(device,train_loader, model, optimizer, epoch, writer,dataset)
        test_BGM(device,test_loader, model, epoch, writer,dataset)
    writer.close()

def BCN_generate_barrier(device, dataset, batch_size=1,checkpoint_path=checkpoint_path,barrier_threshold=0.5):
    if use_full:
        model = fullBGM()
    else:
        model=resizedBGM(dataset)
    model.load_state_dict(torch.load(checkpoint_path + "/bgm_best_f1.model"))
    model = model.to(device)
    model.eval()
    test_loader = torch.utils.data.DataLoader(BoundaryDataset("test",use_full,vid_list_file_tst, num_classes, actions_dict, gt_path, features_path, sample_rate,args.dataset,device),
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True, drop_last=False)
    for index_list, input_data, anchor_xmin, anchor_xmax in test_loader:
        input_data=input_data.to(device)
        BGM_output = model(input_data).detach().cpu().numpy()
        index_list = index_list.numpy()
        barrier=(BGM_output>barrier_threshold)*BGM_output
        columns = ["barrier"]
        for batch_idx, full_idx in enumerate(index_list):
            video_name = test_loader.dataset.list_of_examples[full_idx]
            video_result = barrier[batch_idx]
            video_result=video_result.transpose()
            video_df = pd.DataFrame(video_result, columns=columns)
            video_df.to_csv(bgm_full_result_path + video_name + ".csv", index=False)

def BCN_generate_single_barrier(device,dataset,batch_size=1,checkpoint_path=checkpoint_path,barrier_threshold=0.5,mode='more'):
    if use_full:
        model = fullBGM()
    else:
        model=resizedBGM(dataset)
    model.load_state_dict(torch.load(checkpoint_path + "/bgm_best_f1.model"))
    model = model.to(device)
    model.eval()
    test_loader = torch.utils.data.DataLoader(BoundaryDataset("test",use_full,vid_list_file_tst, num_classes, actions_dict, gt_path, features_path, sample_rate,args.dataset,device),
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True, drop_last=False)

    # for mode = more, we take barriers by >0.8 & (>0.3 & local maximal)
    # for mode = less, we take barriers by >0.5 & local maximal
    if mode=='less':
        for index_list, input_data, anchor_xmin, anchor_xmax in test_loader:
            input_data=input_data.to(device)
            BGM_output = model(input_data).detach().cpu().numpy()
            index_list = index_list.numpy()
            barrier=(BGM_output>barrier_threshold)*BGM_output
            columns = ["barrier"]
            for batch_idx, full_idx in enumerate(index_list):
                video_name = test_loader.dataset.list_of_examples[full_idx]
                video_result = barrier[batch_idx]
                maximum = signal.argrelmax(video_result[0])
                flag=np.array([0]*model.temporal_dim)
                flag[maximum]=1
                video_result=video_result*flag
                video_df = pd.DataFrame(video_result.transpose(), columns=columns)
                video_df.to_csv(bgm_resized_result_path + video_name + ".csv", index=False)

    elif mode=='more':
        for index_list, input_data, anchor_xmin, anchor_xmax in test_loader:
            input_data=input_data.to(device)
            BGM_output = model(input_data).detach().cpu().numpy()
            index_list = index_list.numpy()
            barrier=(BGM_output>0.3)*BGM_output
            high_barrier=(BGM_output>0.8)
            columns = ["barrier"]
            for batch_idx, full_idx in enumerate(index_list):
                video_name = test_loader.dataset.list_of_examples[full_idx]
                video_result = barrier[batch_idx]
                maximum1 = signal.argrelmax(video_result[0])
                maximum2=high_barrier[batch_idx]
                flag=np.array([0]*model.temporal_dim)
                flag[maximum1]=1
                flag=np.clip((flag+maximum2),0,1)
                video_result=video_result*flag
                video_df = pd.DataFrame(video_result.transpose(), columns=columns)
                video_df.to_csv(bgm_resized_result_path + video_name + ".csv", index=False)

def BCN_output_boundary_gt(use_full,batch_size=1):
    test_loader = torch.utils.data.DataLoader(
        BoundaryDataset("gt",use_full, vid_list_file_tst, num_classes, actions_dict, gt_path, features_path, sample_rate,args.dataset,device),
        batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False)
    columns = ["barrier", "xmin", "xmax"]
    for index_list, match_score, anchor_xmin, anchor_xmax in test_loader:
        index_list = index_list.numpy()
        anchor_xmin = np.array([x.numpy()[0] for x in anchor_xmin])
        anchor_xmax = np.array([x.numpy()[0] for x in anchor_xmax])
        for batch_idx, full_idx in enumerate(index_list):
            video = test_loader.dataset.list_of_examples[full_idx]
            barrier = match_score[batch_idx]
            video_result = np.stack((barrier,anchor_xmin, anchor_xmax), axis=1)
            video_df = pd.DataFrame(video_result, columns=columns)
            if use_full:
                video_df.to_csv(bgm_full_gt_path + "gt_" + video + ".csv", index=False)
            else:
                video_df.to_csv(bgm_resized_gt_path + "gt_" + video + ".csv", index=False)

if __name__ == '__main__':
    main()