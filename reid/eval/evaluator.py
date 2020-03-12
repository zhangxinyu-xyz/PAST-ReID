from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from ..evaluation_metrics import cmc, mean_ap
from ..feature_extraction.cnn import extract_cnn_feature
from ..utils.meters import AverageMeter
from ..utils.osutils import save_mat
from ..evaluation_metrics import concat_dict_list
import numpy as np
from .np_distance import compute_dist as np_compute_dist
from .torch_distance import compute_dist as torch_compute_dist
import os


def extract_features(args, model, data_loader, print_freq=10):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    features_numpy = []

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        #if i>=1:
        #    break
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs, args)  # outputs = {'feat': feat}
        outputs = outputs['feat']

        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid
            features_numpy.append(output)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels

def euclidean_dist(x, y):
    """
    Args:
        x: pytorch tensor, with shape [m, d]
        y: pytorch tensor, with shape [n, d]
    Returns:
        dist: pytorch tensor, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def pairwise_distance(query_features, gallery_features, query=None, gallery=None,
                      dist_type='cosine', cos_to_normalize=True):

    x = concat_dict_list(query_features, query)
    y = concat_dict_list(gallery_features, gallery)

    #m, n = x.size(0), y.size(0)
    #x = x.view(m, -1)
    #y = y.view(n, -1)

    if isinstance(x, np.ndarray):
        dist = np_compute_dist(x, y, dist_type=dist_type, cos_to_normalize=cos_to_normalize)
    elif isinstance(x, torch.Tensor):
        dist = torch_compute_dist(x, y, dist_type=dist_type, cos_to_normalize=cos_to_normalize)

    return x, y, dist

def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), name='market1501'):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.2%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),
        'duke': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}

    params = cmc_configs[name]

    cmc_scores = cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)

    print('{} CMC Scores'.format(name))

    for k in cmc_topk:
        print('  top-{:<4}{:12.2%}'.format(k, cmc_scores[k - 1]))

    return cmc_scores[np.array(cmc_topk)-1], mAP

def save_result(q_f, g_f, distmat, savepath, istrain=False):
    q_f = q_f.data.cpu().numpy() if isinstance(q_f, torch.Tensor) else q_f
    g_f = g_f.data.cpu().numpy() if isinstance(g_f, torch.Tensor) else g_f
    distmat = distmat.data.cpu().numpy() if isinstance(distmat, torch.Tensor) else distmat

    if istrain:
        save_mat(os.path.join(savepath, 'results'), 't_f_test.mat', 'features', q_f)
        save_mat(os.path.join(savepath, 'results'), 't_dist_test.mat', 'dist', distmat)
    else:
        save_mat(os.path.join(savepath, 'results'), 'q_f_test.mat', 'features', q_f)
        save_mat(os.path.join(savepath, 'results'), 'g_f_test.mat', 'features', g_f)
        save_mat(os.path.join(savepath, 'results'), 'dist_test.mat', 'dist', distmat)


class Evaluator(object):
    def __init__(self, args, model):
        super(Evaluator, self).__init__()
        self.model = model
        self.args = args

    def evaluate(self, name, query_loader, gallery_loader, query, gallery, savepath=None, issave=False, istrain=False, isevaluate=True):
        print('extracting query features\n')
        query_features, _ = extract_features(self.args, self.model, query_loader, print_freq=self.args.print_freq)
        print('extracting gallery features\n')
        if istrain:
            gallery_features = query_features
        else:
            gallery_features, _ = extract_features(self.args, self.model, gallery_loader, print_freq=self.args.print_freq)

        q_f, g_f, distmat = pairwise_distance(query_features, gallery_features,
                                              query, gallery,
                                              dist_type=self.args.dist_type)

        if issave:
            save_result(q_f, g_f, distmat, savepath, istrain=istrain)

        if isevaluate:
            cmc_scores, mAP = evaluate_all(distmat, query=query, gallery=gallery, name=name)
            return cmc_scores, mAP, q_f, g_f, distmat
        else:
            return None, None, q_f, g_f, distmat
