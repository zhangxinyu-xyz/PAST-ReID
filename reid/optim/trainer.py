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
import copy
from torchvision import transforms

class ReIDTrainer(object):
    def __init__(self, args, cls_params=None, tblogger=None):
        super(ReIDTrainer, self).__init__()

        self.args = args
        self.cls_params = cls_params
        self.tblogger = tblogger
        self.init_log()
        self.init_device()
        if not self.args.evaluate:
            self.init_trainer()
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

    def init_trainer(self):
        self.create_model()
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_lr_scheduler()
        if self.args.idloss_use:
            self.optimizer_promoting = self.create_optimizer_promoting()
            self.lr_scheduler_promoting = self.create_lr_scheduler_promoting(self.args.num_train_loader)
        if self.args.continue_training:
            if self.args.load_optimizer:
                self.load_items(model=True, optimizer=True, lr_scheduler=True)
            else:
                self.load_items(model=True)
        if self.cls_params is not None:
            self.copy_cls_params()

        self.criterion = self.create_criterion()
        self.ckpt_objects = {'model': self.model, 'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}


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

    def copy_cls_params(self):
        for i in range(len(self.model.module.cls_list)):
            self.model.module.cls_list[i].weight = torch.nn.Parameter(self.cls_params[:, i * self.args.embedding_dim: (i + 1) * self.args.embedding_dim])

    def create_optimizer(self):
        # Create self.optimizer, then
        #     recursive_to_device(self.optimizer.state_dict(), self.device)
        # self.optimizer.to(self.device) # One day, there may be this function in official pytorch
        raise NotImplementedError

    def create_optimizer_promoting(self):
        # Create self.optimizer_sup, then
        #     recursive_to_device(self.optimizer_sup.state_dict(), self.device)
        # self.optimizer_sup.to(self.device) # One day, there may be this function in official pytorch
        raise NotImplementedError

    def create_lr_scheduler(self):
        # Create self.lr_scheduler, self.epochs.
        # self.lr_scheduler can be set to None
        raise NotImplementedError

    def create_lr_scheduler_promoting(self, num_train_loader):
        # Create self.lr_scheduler_sup, self.epochs.
        # self.lr_scheduler_sup can be set to None
        raise NotImplementedError

    def create_criterion(self):
        # loss = self.criterion(batch, pred)
        raise NotImplementedError

    def set_model_to_train_mode(self):
        # Default is self.model.train()
        raise NotImplementedError

    def set_model_to_test_mode(self):
        # Default is self.model.train()
        raise NotImplementedError

    def may_save_ckpt(self, name=None):
        save_checkpoint(self.ckpt_objects, self.args.save_dir, name=name)

    def update_cls_params(self, model, new_cls_params):
        params = model.state_dict()
        self.model = copy_state_dict(self.model, params)

        self.cls_params = copy.deepcopy(new_cls_params)
        self.model.module.cls_params = copy.deepcopy(new_cls_params)

        self.num_classes = copy.deepcopy(self.cls_params.size(0))
        self.model.module.num_classes = copy.deepcopy(self.cls_params.size(0))

        if self.args.pool_type == 'PCBPool_nine':
            self.model.module.cls_list = copy.deepcopy(nn.ModuleList(
                    [nn.Linear(self.args.embedding_dim, self.num_classes) for _ in range(self.args.num_parts + 3)]))
        else:
            self.model.module.cls_list = copy.deepcopy(nn.ModuleList(
                [nn.Linear(self.args.embedding_dim, self.num_classes) for _ in range(self.args.num_parts)]))

        for i in range(len(self.model.module.cls_list)):
            self.model.module.cls_list[i].weight = torch.nn.Parameter(self.cls_params[:, i * self.args.embedding_dim: (i + 1) * self.args.embedding_dim])

        self.cls_list = copy.deepcopy(self.model.module.cls_list)

        return self.model.cuda()

    def train_promoting(self, iteration, epoch, data_loader, print_freq=1, cls_list=None, stage='Promoting Stage'):
        self.set_model_to_train_mode()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()
        len_dataloader = len(data_loader)

        for batch_idx, inputs in enumerate(data_loader):
            # if epoch == 0:
            #if batch_idx >= 1:
            #    break
            data_time.update(time.time() - end)

            if self.lr_scheduler_promoting is not None:
                if self.lr_scheduler_promoting.last_epoch == -1 and self.args.start_epochs != 0:
                    self.lr_scheduler_promoting.step(epoch=self.args.start_iters * self.args.epochs)
                else:
                    self.lr_scheduler_promoting.step()
            self.optimizer_promoting.zero_grad()

            inputs, targets = self._parse_data(inputs)
            out_dict, loss, prec = self._forward(inputs, targets=targets, cls_list=cls_list, tri_loss_type='IDL')

            # ===================================================================================
            losses.update(loss.data, targets.size(0))
            precisions.update(prec, targets.size(0))

            if isinstance(loss, torch.Tensor):
                loss.backward()

            self.optimizer_promoting.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % print_freq == 0 or (batch_idx + 1) == len_dataloader:
                print_log(stage, iteration, self.args.iters, epoch, batch_idx, len_dataloader,
                          batch_time, data_time, losses, precisions,
                          self.optimizer_promoting.param_groups[0]['lr'])

        self.score = {'loss': losses, 'prec': precisions}

    def train_conservative(self, iteration, epoch, data_loader_CTL, ori_data_loader_RTL, print_freq=1, cls_list=None, stage='Conservative Stage'):
        self.set_model_to_train_mode()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        #if self.args.idloss_use:
        precisions = AverageMeter()
        end = time.time()

        len_dataloader = len(data_loader_CTL)

        data_loader_RTL = ori_data_loader_RTL.__iter__()
        del ori_data_loader_RTL

        for batch_idx, inputs in enumerate(data_loader_CTL):
            #if batch_idx >= 1:
            #    break
            data_time.update(time.time() - end)

            if self.lr_scheduler is not None:
                if self.lr_scheduler.last_epoch == -1 and self.args.start_epochs != 0:
                    self.lr_scheduler.step(epoch=self.args.start_epochs*self.args.num_train_loader)
                else:
                    self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            if self.args.tri_sampler_type == 'CTL_RTL':
                inputs, targets = self._parse_data(inputs, tri_loss_type='CTL')
                out_dict_CTL, loss_CTL, prec_CTL \
                    = self._forward(inputs, targets=targets, tri_loss_type='CTL')
                
                inputs = data_loader_RTL.__next__()
                inputs, targets = self._parse_data(inputs, tri_loss_type='RTL')
                out_dict_RTL, loss_RTL, prec_RTL \
                    = self._forward(inputs, targets=targets, cls_list=cls_list, tri_loss_type='RTL')

                loss = loss_CTL + loss_RTL
                prec = prec_CTL + prec_RTL

            elif self.args.triloss_use and self.args.tri_sampler_type == 'RTL':
                inputs, targets = self._parse_data(inputs)
                out_dict, loss, prec, id_loss = self._forward(inputs, targets=targets, cls_list=cls_list)
            else:
                inputs, targets = self._parse_data(inputs)
                out_dict, loss, prec, id_loss = self._forward(inputs, targets=targets, cls_list=cls_list)

            # ===================================================================================
            losses.update(loss.data, targets.size(0))
            precisions.update(prec, targets.size(0))

            if isinstance(loss, torch.Tensor):
                loss.backward()

            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % print_freq == 0 or (batch_idx + 1) == len_dataloader:
                print_log(stage, iteration, self.args.iters, epoch, batch_idx, len_dataloader,
                          batch_time, data_time, losses, precisions,
                          self.optimizer.param_groups[0]['lr'])

        self.score = {'loss': losses, 'prec':precisions}

    def test(self):
        from reid.eval.evaluator import Evaluator
        evaluator = Evaluator(self.args, self.model)
        return evaluator

    def batch_images(self, tag, data_set, global_step):
        self.tblogger.add_images(os.path.join(self.args.dbscan_type, tag + '_batch'), data_set, global_step)

    def _parse_data(self, inputs, tri_loss_type=None):
        raise NotImplementedError

    def _forward(self, inputs, targets, cls_list=None, tri_loss_type=None):
        raise NotImplementedError

