from __future__ import absolute_import

from .classification import accuracy
from .ranking import cmc, mean_ap
import numpy as np
import itertools
import torch

__all__ = [
    'accuracy',
    'cmc',
    'mean_ap',
]


def concat_dict_list(dict_list, names):

    if isinstance(dict_list[names[0][0]], list):
        ret_dict = list(itertools.chain.from_iterable([dict_list[f] for f, _, _ in names]))
    elif isinstance(dict_list[names[0][0]], np.ndarray):
        ret_dict = np.concatenate([dict_list[f][np.newaxis, ...] for f, _, _ in names])
    elif isinstance(dict_list[names[0][0]], torch.Tensor):
        ret_dict = torch.cat([dict_list[f].unsqueeze(0) for f, _, _ in names], 0)
    else:
        raise NotImplementedError
    return ret_dict
