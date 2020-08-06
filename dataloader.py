import torch
import numpy as np
from torch.utils import data


class VideoBoundaryDataset(data.Dataset):
    def __init__(self, vid_list_file, num_classes, actions_dict, gt_path, features_path, sample_rate,dataset,device,bd_ratio=0.05):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.device = device
        self.boundary_ratio = bd_ratio
        self.dataset = dataset
        if self.dataset=='50salads':
            self.bg_class=[17,18]
        elif self.dataset=='gtea':
            self.bg_class = [10]
        elif self.dataset=='breakfast':
            self.bg_class = [0]

        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

    def __getitem__(self, index):
        '''
        :return: mask[batch_size, num_classes, max(length_of_sequences)]
        '''
        feature_tensor, target_tensor, mask,anchor_xmin,anchor_xmax=self._get_base_data(index)
        match_score_start, match_score_end = self._get_train_label(index, target_tensor, anchor_xmin, anchor_xmax)
        match_score = torch.cat((match_score_start.unsqueeze(0), match_score_end.unsqueeze(0)), 0)
        match_score,_ = torch.max(match_score, 0)#.values()
        return feature_tensor, target_tensor, mask, match_score

    def __len__(self):
        return len(self.list_of_examples)

    def _get_base_data(self,index):
        features = np.load(self.features_path + self.list_of_examples[index].split('.')[0] + '.npy')
        file_ptr = open(self.gt_path + self.list_of_examples[index], 'r')
        content = file_ptr.read().split('\n')[:-1]  # read ground truth
        # initialize and produce gt vector
        classes = np.zeros(min(np.shape(features)[1], len(content)))
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]

        # sample information by skipping each sample_rate frames
        features = features[:, ::self.sample_rate]
        target = classes[::self.sample_rate]

        # create pytorch tensor
        feature_tensor = torch.tensor(features, dtype=torch.float)
        target_tensor = torch.tensor(target, dtype=torch.long)
        mask = torch.ones(self.num_classes, np.shape(target)[0], dtype=torch.float)

        total_frame = target_tensor.size()[0]
        temporal_scale = total_frame
        temporal_gap = 1.0 / temporal_scale
        anchor_xmin = [temporal_gap * i for i in range(temporal_scale)]
        anchor_xmax = [temporal_gap * i for i in range(1, temporal_scale + 1)]
        return feature_tensor, target_tensor, mask, anchor_xmin, anchor_xmax

    def _get_train_label(self, index, target_tensor, anchor_xmin, anchor_xmax):
        total_frame = target_tensor.size()[0]
        temporal_scale = total_frame
        temporal_gap = 1.0 / temporal_scale
        gt_label, gt_starts, gt_ends = self._get_labels_start_end_time(target_tensor, self.bg_class)  # original length
        gt_label, gt_starts, gt_ends = np.array(gt_label), np.array(gt_starts), np.array(gt_ends)
        gt_starts, gt_ends = gt_starts.astype(np.float), gt_ends.astype(np.float)
        gt_starts, gt_ends = gt_starts / total_frame, gt_ends / total_frame  # length to 0~1

        gt_lens = gt_ends - gt_starts
        gt_len_small = np.maximum(temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_starts - gt_len_small / 2, gt_starts + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_ends - gt_len_small / 2, gt_ends + gt_len_small / 2), axis=1)

        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        return match_score_start, match_score_end

    def _ioa_with_anchors(self,anchors_min,anchors_max,box_min,box_max):
        len_anchors=anchors_max-anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.0)
        scores = np.divide(inter_len, len_anchors)
        return scores

    def _get_labels_start_end_time(self,target_tensor, bg_class):
        labels = []
        starts = []
        ends = []
        target=target_tensor.numpy()
        last_label = target[0]
        if target[0] not in bg_class:
            labels.append(target[0])
            starts.append(0)

        for i in range(np.shape(target)[0]):
            if target[i] != last_label:
                if target[i] not in bg_class:
                    labels.append(target[i])
                    starts.append(i)
                if last_label not in bg_class:
                    ends.append(i)
                last_label = target[i]

        if last_label not in bg_class:
            ends.append(np.shape(target)[0]-1)
        return labels, starts, ends




