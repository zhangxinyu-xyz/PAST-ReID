from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import os
from ..osutils import load_mat
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)
import random
import copy
import scipy.io as scio
from ..osutils import save_mat


##################### CTL sampler ######################
class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances=4, savepath=None):  #epoch=-1,
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        # self.epoch = epoch
        self.savepath = savepath
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)  #
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        save_mat(os.path.join(self.savepath, 'samples'), 'CTL_samples.mat', 'samples', final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

##################### RTL sampler ######################
class SoftMarginTripletSampler(Sampler):
    def __init__(self, data_source, dist=None, k=20, data_indices=None):
        assert dist is not None, "There is no distmat to generate triplets"
        self.data_source = data_source
        self.dist = dist
        self.data_indices = data_indices
        if self.data_indices is not None:
            self.num_samples = len(self.data_indices)
        else:
            self.num_samples = len(data_source)
        if k*2 >= self.num_samples:
            k = int(self.num_samples * 0.5)
        self.k = k

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            if self.data_indices is not None:
                indice = self.data_indices[i]
                index = np.argsort(self.dist[indice][self.data_indices])
            else:
                index = np.argsort(self.dist[i])

            pos_ind = int(np.random.random() * self.k)   # pos_ind: random index from [0:k]
            pos_pos = index[pos_ind]   # pos_pos: position of distmat if pos_ind

            neg_ind = int(np.random.random() * self.k) + self.k  # neg_ind: random index from [k:2k]
            neg_pos = index[neg_ind]   # neg_pos: position of distmat if neg_ind

            ret.append([i, pos_ind, pos_pos, neg_ind, neg_pos])

        return iter(ret)

##################### CTL_RTL sampler ######################
class RandomIdSoftmarginTripletSampler(Sampler):
    def __init__(self, data_source, dist=None, k=20, data_indices=None, savepath=None):  #epoch=-1,
        assert dist is not None, "There is no distmat to generate triplets"
        self.data_source = data_source
        self.dist = dist
        self.data_indices = np.array(data_indices)
        self.savepath = savepath
        self.dist = dist[self.data_indices][:, self.data_indices]
        self.num_samples = len(self.data_indices)
        if k * 2 >= self.num_samples:
            k = int(self.num_samples * 0.5)
        self.k = k

    def __iter__(self):
        self.select_indices = \
            scio.loadmat(os.path.join(self.savepath, 'samples', 'CTL_samples.mat'))['samples'][0]
        ret = []
        for i in self.select_indices:
            index = np.argsort(self.dist[i])

            pos_ind = int(np.random.random() * self.k)   # pos_ind: random index from [0:k]
            pos_pos = index[pos_ind]   # pos_pos: position of sorted distmat if pos_ind

            neg_ind = int(np.random.random() * self.k) + self.k  # neg_ind: random index from [k:2k]
            neg_pos = index[neg_ind]   # neg_pos: position of distmat if neg_ind

            ret.append([i, pos_ind, pos_pos, neg_ind, neg_pos])

        save_mat(os.path.join(self.savepath, 'samples'), 'RTL_samples.mat', 'samples', ret)
        return iter(ret)

    def __len__(self):
        return self.num_samples



