from __future__ import print_function, absolute_import
import time

import torch
from torch import nn

from ..utils.meters import AverageMeter
from ..utils.serialization import get_default_device
from ..utils.logging import ReDirectSTD, print_log
from ..utils.logging import time_str as t_str
from ..utils.serialization import save_checkpoint, load_checkpoint, copy_state_dict
import os
from torchvision import transforms

class ReIDTrainer(object):
    def __init__(self, args, cls_params=None):
        super(ReIDTrainer, self).__init__()

        self.args = args
        assert args.evaluate, "This version of code is only for test. We will release the training code later."
        self.cls_params = cls_params
        self.init_log()
        self.init_device()
        self.init_eval()
        
        self.to_pil_image = transforms.ToPILImage()

    def init_log(self):
        args = self.args
        # Redirect logs to both console and file.
        time_str = t_str()
        ReDirectSTD(os.path.join(args.log_dir, 'stdout_{}.txt'.format(time_str)), 'stdout', True)
        ReDirectSTD(os.path.join(args.log_dir, 'stderr_{}.txt'.format(time_str)), 'stderr', True)
        print('=> Experiment Output Directory: {}'.format(args.log_dir))
        print('[PYTORCH VERSION]:', torch.__version__)

    def init_device(self):
        self.device = get_default_device()
        self.args.device = self.device

    def init_eval(self):
        if not hasattr(self, 'model'):
            self.create_model()
        if self.args.evaluate:
            self.load_items(model=True)

    def load_items(self, model=False, optimizer=False, lr_scheduler=False):
        """To allow flexible multi-stage training."""
        objects = {}
        if model:
            objects['model'] = self.model
        if optimizer:
            objects['optimizer'] = self.optimizer
        if lr_scheduler:
            objects['lr_scheduler'] = self.lr_scheduler
        load_checkpoint(objects, self.args.resume_file, strict=False)

    def set_model_to_train_mode(self):
        # Default is self.model.train()
        raise NotImplementedError

    def set_model_to_test_mode(self):
        # Default is self.model.train()
        raise NotImplementedError

    def may_save_ckpt(self, name=None):
        save_checkpoint(self.ckpt_objects, self.args.save_dir, name=name)

    def test(self):
        from reid.eval.evaluator import Evaluator
        evaluator = Evaluator(self.args, self.model)
        return evaluator


