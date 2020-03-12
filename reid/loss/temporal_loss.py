from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from ..eval.torch_distance import compute_dist, compute_correspond_dist
import numpy as np

class UnSupervisedTripletLoss(object):
    def __init__(self, args, tb_writer=None):
        super(UnSupervisedTripletLoss, self).__init__()
        self.args = args
        self.tri_loss_obj = _TripletLoss(args, margin=args.margin)

    def calculate(self, a_feat, p_feat, n_feat, p_n_index):
        """
        Args:
            feat: pytorch tensor, shape [N, C]
            labels: pytorch LongTensor, with shape [N]
            hard_type: can be dynamically set to different types during training, for hybrid or curriculum learning
        Returns:
            loss: pytorch scalar
            ==================
            For Debugging, etc
            ==================
            dist_ap: pytorch tensor, distance(anchor, positive); shape [N]
            dist_an: pytorch tensor, distance(anchor, negative); shape [N]
            dist_mat: pytorch tensor, pairwise euclidean distance; shape [N, N]
        """
        dist_ap = compute_correspond_dist(a_feat, p_feat, self.args.dist_type)
        dist_an = compute_correspond_dist(a_feat, n_feat, self.args.dist_type)
        softmargin = (p_n_index[0] - p_n_index[2]).abs().float() / self.args.k_nearest

        loss = self.tri_loss_obj(dist_ap, dist_an, softmargin=softmargin)

        return {'loss': loss, 'dist_ap': dist_ap, 'dist_an': dist_an}

    def __call__(self, a_pred, p_pred, n_pred, p_n_index=None):
        # NOTE: Here is only a trial implementation for PCB-6P*
        # Calculation Part similar
        assert p_n_index is not None, "There is no index of the positive and negative to produce softmargin"
        loss = 0

        if self.args.triloss_part == 3:
            res1 = self.calculate(torch.cat(a_pred['feat_list'][:6], 1), torch.cat(p_pred['feat_list'][:6], 1),
                                  torch.cat(n_pred['feat_list'][:6], 1), p_n_index)
            res2 = self.calculate(torch.cat(a_pred['feat_list'][6:8], 1), torch.cat(p_pred['feat_list'][6:8], 1),
                                  torch.cat(n_pred['feat_list'][6:8], 1), p_n_index)
            res3 = self.calculate(a_pred['feat_list'][8], p_pred['feat_list'][8],
                                  n_pred['feat_list'][8], p_n_index)

            loss = res1['loss'] + res2['loss'] + res3['loss']
        elif self.args.triloss_part == 4:
            res1 = self.calculate(torch.cat(a_pred['feat_list'][:6], 1), torch.cat(p_pred['feat_list'][:6], 1),
                                  torch.cat(n_pred['feat_list'][:6], 1), p_n_index)
            res2 = self.calculate(a_pred['feat_list'][6], p_pred['feat_list'][6],
                                  n_pred['feat_list'][6], p_n_index)
            res3 = self.calculate(a_pred['feat_list'][7], p_pred['feat_list'][7],
                                  n_pred['feat_list'][7], p_n_index)
            res4 = self.calculate(a_pred['feat_list'][8], p_pred['feat_list'][8],
                                  n_pred['feat_list'][8], p_n_index)

            loss = res1['loss'] + res2['loss'] + res3['loss'] + res4['loss']
        elif self.args.triloss_part == 1:
            res = self.calculate(torch.cat(a_pred['feat_list'], 1), torch.cat(p_pred['feat_list'], 1),
                                torch.cat(n_pred['feat_list'], 1), p_n_index)
            loss += res['loss']
        else:
            for i in range(len(a_pred['feat_list'])):
                res = self.calculate(a_pred['feat_list'][i], p_pred['feat_list'][i], n_pred['feat_list'][i], p_n_index)
                loss += res['loss']
        #loss = self.calculate(torch.cat(pred['feat_list'], 1), target, hard_type=hard_type)['loss']
        # Scale by loss weight
        #loss *= self.args.triloss_weight

        if self.args.triloss_mean:
            loss = loss / len(a_pred['feat_list'])

        return {'loss': loss}