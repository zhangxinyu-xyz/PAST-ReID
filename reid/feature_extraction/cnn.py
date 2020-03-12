from __future__ import absolute_import
from torch.autograd import Variable
from ..utils import to_torch
import torch
import numpy as np

def torch_normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def np_normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)

def extract_cnn_feature(model, inputs, args):
    model.eval()
    with torch.no_grad():
        #in_dict = recursive_to_device(in_dict, cfg.device)
        inputs = to_torch(inputs)
        inputs = Variable(inputs, volatile=True)
        out_dict = model(inputs, forward_type=args.forward_type)
        out_dict['feat_list'] = [torch_normalize(f) for f in out_dict['feat_list']]
        feat = torch.cat(out_dict['feat_list'], 1)
        feat = feat.cpu().numpy()
        #feat = [f.cpu() for f in out_dict['feat_list']]
        ret_dict = {'feat': feat}
    return ret_dict

def extract_cnn_part_feature(model, inputs, args):
    model.eval()
    with torch.no_grad():
        #in_dict = recursive_to_device(in_dict, cfg.device)
        inputs = to_torch(inputs)
        inputs = Variable(inputs, volatile=True)
        out_dict = model(inputs, forward_type=args.forward_type)
        out_dict['feat_list'] = [torch_normalize(f) for f in out_dict['feat_list']]
        feat = [f.cpu().numpy() for f in out_dict['feat_list']]
        feat = np.array(feat)
        #feat = [f.cpu() for f in out_dict['feat_list']]
        ret_dict = {'feat': feat}
    return ret_dict


