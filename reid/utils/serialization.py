from __future__ import print_function, absolute_import
import json
import shutil

import torch
from torch.nn import Parameter
from torch.nn.parallel import DataParallel
import os

from .osutils import mkdir_if_missing


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(os.path.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(objects, ckpt_path, name=None):
    """Save state_dict's of modules/optimizers/lr_schedulers to file.
    Args:
        objects: A dict, which members are either
            torch.nn.optimizer
            or torch.nn.Module
            or torch.optim.lr_scheduler._LRScheduler
            or None
        name: The model name.
        ckpt_file: The file path.
    Note:
        torch.save() reserves device type and id of tensors to save, so when
        loading ckpt, you have to inform torch.load() to load these tensors to
        cpu or your desired gpu, if you change devices.
    """
    state_dicts = {name: obj.state_dict() for name, obj in objects.items() if obj is not None}
    ckpt = dict(state_dicts=state_dicts)
    mkdir_if_missing(ckpt_path)
    ckpt_file = os.path.join(ckpt_path, name + '.pth.tar')
    torch.save(ckpt, ckpt_file)
    msg = '=> Checkpoint Saved to {}'.format(ckpt_file)
    print(msg)

def load_state_dict(model, src_state_dict, fold_bnt=True):
    """Copy parameters and buffers from `src_state_dict` into `model` and its
    descendants. The `src_state_dict.keys()` NEED NOT exactly match
    `model.state_dict().keys()`. For dict key mismatch, just
    skip it; for copying error, just output warnings and proceed.
    Arguments:
        model: A torch.nn.Module object.
        src_state_dict (dict): A dict containing parameters and persistent buffers.
    Note:
        This is modified from torch.nn.modules.module.load_state_dict(), to make
        the warnings and errors more detailed.
    """
    from torch.nn import Parameter

    dest_state_dict = model.state_dict()
    for name, param in src_state_dict.items():
        if name not in dest_state_dict:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            dest_state_dict[name].copy_(param)
        except Exception:
            print("Warning: Error occurs when copying '{}'".format(name))

    # New version of BN has buffer `num_batches_tracked`, which is not used
    # for normal BN, so we fold all these missing keys into one line
    def _fold_nbt(keys):
        nbt_keys = [s for s in keys if s.endswith('.num_batches_tracked')]
        if len(nbt_keys) > 0:
            keys = [s for s in keys if not s.endswith('.num_batches_tracked')] + ['num_batches_tracked  x{}'.format(len(nbt_keys))]
        return keys

    src_missing = set(dest_state_dict.keys()) - set(src_state_dict.keys())
    if len(src_missing) > 0:
        print("Keys not found in source state_dict: ")
        if fold_bnt:
            src_missing = _fold_nbt(src_missing)
        for n in src_missing:
            print('\t', n)

    dest_missing = set(src_state_dict.keys()) - set(dest_state_dict.keys())
    if len(dest_missing) > 0:
        print("Keys not found in destination state_dict: ")
        if fold_bnt:
            dest_missing = _fold_nbt(dest_missing)
        for n in dest_missing:
            print('\t', n)

def load_checkpoint(objects, ckpt_file, strict=True):
    """Load state_dict's of modules/optimizers/lr_schedulers from file.
    Args:
        objects: A dict, which values are either
            torch.nn.optimizer
            or torch.nn.Module
            or torch.optim.lr_scheduler._LRScheduler
            or None
        ckpt_file: The file path.
    """
    assert os.path.exists(ckpt_file), "ckpt_file {} does not exist!".format(ckpt_file)
    assert os.path.isfile(ckpt_file), "ckpt_file {} is not file!".format(ckpt_file)
    ckpt = torch.load(ckpt_file, map_location=(lambda storage, loc: storage))
    for name, obj in objects.items():
        if obj is not None:
            # Only nn.Module.load_state_dict has this keyword argument
            if not isinstance(obj, torch.nn.Module) or strict:
                obj.load_state_dict(ckpt['state_dicts'][name])
            else:
                load_state_dict(obj, ckpt['state_dicts'][name])
    objects_str = ', '.join(objects.keys())
    msg = '=> Loaded [{}] from {}\n'.format(objects_str, ckpt_file)
    print(msg)

def copy_state_dict(model, src_state_dict, strip=None, fold_bnt=True):
    
    dest_state_dict = model.state_dict()
    for name, param in src_state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in dest_state_dict:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != dest_state_dict[name].size():
            print('mismatch:', name, param.size(), dest_state_dict[name].size())
            continue
        try:
            dest_state_dict[name].copy_(param)
        except Exception:
            print("Warning: Error occurs when copying '{}'".format(name))

    # New version of BN has buffer `num_batches_tracked`, which is not used
    # for normal BN, so we fold all these missing keys into one line
    def _fold_nbt(keys):
        nbt_keys = [s for s in keys if s.endswith('.num_batches_tracked')]
        if len(nbt_keys) > 0:
            keys = [s for s in keys if not s.endswith('.num_batches_tracked')] + ['num_batches_tracked  x{}'.format(len(nbt_keys))]
        return keys

    src_missing = set(dest_state_dict.keys()) - set(src_state_dict.keys())
    if len(src_missing) > 0:
        print("Keys not found in source state_dict: ")
        if fold_bnt:
            src_missing = _fold_nbt(src_missing)
        for n in src_missing:
            print('\t', n)

    dest_missing = set(src_state_dict.keys()) - set(dest_state_dict.keys())
    if len(dest_missing) > 0:
        print("Keys not found in destination state_dict: ")
        if fold_bnt:
            dest_missing = _fold_nbt(dest_missing)
        for n in dest_missing:
            print('\t', n)

    return model

def get_default_device():
    """Get default device for `*.to(device)`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def recursive_to_device(input, device):
    """NOTE: If input is dict/list/tuple, it is changed in place."""
    if isinstance(input, torch.Tensor):
        # print('=> IS torch.Tensor')
        # print('=> input.device before to_device: {}'.format(input.device))
        input = input.to(device)
        # print('=> input.device after to_device: {}'.format(input.device))
    elif isinstance(input, dict):
        for k, v in input.items():
            input[k] = recursive_to_device(v, device)
    elif isinstance(input, (list, tuple)):
        input = [recursive_to_device(v, device) for v in input]
    return input

