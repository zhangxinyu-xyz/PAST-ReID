from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from ..eval.torch_distance import compute_dist, compute_correspond_dist

class _TripletLoss(object):
    """Reference:
        https://github.com/Cysu/open-reid
        In Defense of the Triplet Loss for Person Re-Identification
    """
    def __init__(self, args, margin=None, name=None, tri_sampler_type='CTL'):
        self.margin = margin
        self.args = args
        self.name = name
        self.tri_sampler_type = tri_sampler_type
        if margin is not None:
            if self.tri_sampler_type == 'CTL':
                self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
            elif self.tri_sampler_type == 'RTL':
                self.ranking_loss = SoftMarginTriplet(margin=self.margin)
            elif self.tri_sampler_type == 'CTL_RTL':
                if '_CTL' in name:
                    self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
                if '_RTL' in name:
                    self.ranking_loss = SoftMarginTriplet(margin=self.margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, dist_ap, dist_an, softmargin=None):
        """
        Args:
          dist_ap: pytorch tensor, distance between anchor and positive sample, shape [N]
          dist_an: pytorch tensor, distance between anchor and negative sample, shape [N]
        Returns:
          loss: pytorch scalar
        """
        y = torch.ones_like(dist_ap)
        if self.margin is not None:
            if self.tri_sampler_type == 'CTL':
                loss = self.ranking_loss(dist_an, dist_ap, y)
                return loss
            elif self.tri_sampler_type == 'RTL':
            #elif 'softmargin' in self.args.tri_sampler_type:
                loss = self.ranking_loss(dist_ap, dist_an, softmargin)
                return loss
            elif self.tri_sampler_type == 'CTL_RTL':
                if '_CTL' in self.name:
                    loss = self.ranking_loss(dist_an, dist_ap, y)
                if '_RTL' in self.name:
                    loss = self.ranking_loss(dist_ap, dist_an, softmargin)
                return loss
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss

class SoftMarginTriplet(_Loss):
    __constants__ = ['reduction']
    '''
    inputs `x1`, `x2`, two 1D mini-batch `Tensor`s,
    and a label 1D mini-batch tensor `y` with values (`1` or `-1`).

    If `y == 1` then it assumed the first input should be ranked higher
    (have a larger value) than the second input, and vice-versa for `y == -1`.

    The loss function for each sample in the mini-batch is:
    
    loss(x, y) = max(0, -y * (x1 - x2) + margin)  
    
    reduction='elementwise_mean'|'none'|'sum'
    '''
    def __init__(self, margin=0., size_average=None, reduce=None, reduction='elementwise_mean'):
        super(SoftMarginTriplet, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, dist_ap, dist_an, softmargin):
        loss = F.relu(dist_ap - dist_an + softmargin * self.margin)
        if self.reduction == 'elementwise_mean':
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def construct_triplets(dist_mat, labels, hard_type='tri_hard'):
    """Construct triplets inside a batch.
    Args:
        dist_mat: pytorch tensor, pair wise distance between samples, shape [N, N]
        labels: pytorch LongTensor, with shape [N]
    Returns:
        dist_ap: pytorch tensor, distance(anchor, positive); shape [M]
        dist_an: pytorch tensor, distance(anchor, negative); shape [M]
    NOTE: Only consider PK batch, so we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_self = labels.new().resize_(N, N).copy_(torch.eye(N)).byte()
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    K = is_pos.sum(1)[0]
    P = int(N / K)
    assert P * K == N, "P * K = {}, N = {}".format(P * K, N)
    is_pos = ~ is_self & is_pos
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    if hard_type == 'semi':
        dist_ap = dist_mat[is_pos].contiguous().view(-1)
        dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        dist_an = dist_an.expand(N, K - 1).contiguous().view(-1)
    elif hard_type == 'all':
        dist_ap = dist_mat[is_pos].contiguous().view(N, K - 1).unsqueeze(-1).expand(N, K - 1, P * K - K).contiguous().view(-1)
        dist_an = dist_mat[is_neg].contiguous().view(N, P * K - K).unsqueeze(1).expand(N, K - 1, P * K - K).contiguous().view(-1)
    elif hard_type == 'tri_hard':
        # `dist_ap` means distance(anchor, positive); both `dist_ap` and `relative_p_inds` with shape [N]
        dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=False)
        # `dist_an` means distance(anchor, negative); both `dist_an` and `relative_n_inds` with shape [N]
        dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=False)
    else:
        raise NotImplementedError

    # print("dist_ap.size() {}, dist_an.size() {}, N {}, P {}, K {}".format(dist_ap.size(), dist_an.size(), N, P, K))
    assert dist_ap.size() == dist_an.size(), "dist_ap.size() {}, dist_an.size() {}".format(dist_ap.size(), dist_an.size())
    return dist_ap, dist_an


class CTLTripletLoss(object):
    def __init__(self, args, name=None, tri_sampler_type='CTL'):
        super(CTLTripletLoss, self).__init__()
        self.args = args
        self.tri_loss_obj = _TripletLoss(args, margin=args.margin, name=name, tri_sampler_type=tri_sampler_type)

    def calculate(self, feat, labels, hard_type=None):
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
        dist_mat = compute_dist(feat, feat, dist_type=self.args.dist_type)
        if hard_type is None:
            hard_type = self.args.hard_type
        dist_ap, dist_an = construct_triplets(dist_mat, labels, hard_type=hard_type)
        loss = self.tri_loss_obj(dist_ap, dist_an)
        if self.args.norm_by_num_of_effective_triplets:
            sm = (dist_an > dist_ap + self.args.margin).float().mean().item()
            loss *= 1. / (1 - sm + 1e-8)
        return {'loss': loss, 'dist_ap': dist_ap, 'dist_an': dist_an}

    def __call__(self, target, pred, step=0, hard_type=None):
        # NOTE: Here is only a trial implementation for PCB-6P*

        # Calculation Part similar
        loss = 0

        for i in range(len(pred['feat_list'])):
            res = self.calculate(pred['feat_list'][i], target, hard_type=hard_type)
            loss += res['loss']

        return {'loss': loss}


class RTLTripletLoss(object):
    def __init__(self, args, name=None, tri_sampler_type='RTL'):
        super(RTLTripletLoss, self).__init__()
        self.args = args
        self.tri_loss_obj = _TripletLoss(args, margin=args.margin, name=name, tri_sampler_type=tri_sampler_type)

    def calculate(self, a_feat, p_feat, n_feat, p_n_index):
        """
        Args:
            a_feat: pytorch tensor, anchor features; shape [N, C]
            p_feat: pytorch tensor, positive features; shape [N, C]
            n_feat: pytorch tensor, negative features; shape [N, C]
            p_n_index: pytorch tensor, ranking relative distance(positive, negative); shape [N]
        Returns:
            loss: pytorch scalar
        """
        dist_ap = compute_correspond_dist(a_feat, p_feat, self.args.dist_type)
        dist_an = compute_correspond_dist(a_feat, n_feat, self.args.dist_type)
        softmargin = (p_n_index[1] - p_n_index[3]).abs().float() / (self.args.k_nearest * 2)

        loss = self.tri_loss_obj(dist_ap, dist_an, softmargin)

        return {'loss': loss, 'dist_ap': dist_ap, 'dist_an': dist_an}

    def __call__(self, a_pred, p_pred, n_pred, p_n_index=None):
        # NOTE: Here is only a trial implementation for PCB-6P*
        # Calculation Part similar
        assert p_n_index is not None, "There is no index of the positive and negative to produce softmargin"
        loss = 0

        for i in range(len(a_pred['feat_list'])):
                res = self.calculate(a_pred['feat_list'][i], p_pred['feat_list'][i], n_pred['feat_list'][i], p_n_index)
                loss += res['loss']

        if self.args.triloss_mean:
            loss = loss / len(a_pred['feat_list'])

        return {'loss': loss}
