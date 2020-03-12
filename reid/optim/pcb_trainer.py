from __future__ import print_function
from collections import OrderedDict
from copy import deepcopy
import torch
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from .trainer import ReIDTrainer

from itertools import chain
from .optimizer import create_optimizer
from ..utils.serialization import recursive_to_device
from torch.autograd import Variable
from ..loss.id_loss import IDLoss
from ..loss.triplet_loss import RTLTripletLoss, CTLTripletLoss
from ..models.model import Model
from torch import nn
import copy


class PCBTrainer(ReIDTrainer):

    def set_model_to_train_mode(self):
        self.model.module.set_train_mode(fix_ft_layers=self.args.phase == 'fix_finetune_layers')

    def set_model_to_test_mode(self):
        self.model.module.set_test_mode()

    def create_model(self):
        self.model = Model(deepcopy(self.args), self.cls_params)
        self.model = nn.DataParallel(self.model).cuda()

    def create_optimizer(self):
        # Optimizer
        ft_params, new_params = self.get_ft_and_new_params()
        if self.args.phase == 'scratch':
            assert len(new_params) > 0, "No new params to pretrain!"
            param_groups = [{'params': ft_params, 'lr': self.args.new_params_lr}]
            param_groups += [{'params': new_params, 'lr': self.args.new_params_lr}]
        elif self.args.phase == 'normal':
            param_groups = [{'params': ft_params, 'lr': self.args.ft_lr}]
            # Some model may not have new params
            if len(new_params) > 0:
                param_groups += [{'params': new_params, 'lr': self.args.new_params_lr}]
        elif self.args.phase == 'finetune':
            param_groups = [{'params': ft_params, 'lr': self.args.ft_lr}]
            param_groups += [{'params': new_params, 'lr': self.args.ft_lr}]
        else:
            param_groups = [{'params': ft_params, 'lr': 0.0}]
            param_groups += [{'params': new_params, 'lr': self.args.ft_lr}]

        self.optimizer = create_optimizer(param_groups, self.args)
        recursive_to_device(self.optimizer.state_dict(), self.device)
        return self.optimizer

    def create_optimizer_promoting(self):
        # Optimizer
        ft_modules = [self.model.module.backbone]
        ft_modules += [self.model.module.em_list]
        if hasattr(self.model.module, 'cls_list'):
            new_modules = [self.model.module.cls_list]
        else:
            new_modules = None

        ft_params = list(chain.from_iterable([list(m.parameters()) for m in ft_modules]))
        new_params = list(chain.from_iterable([list(m.parameters()) for m in new_modules]))

        if self.args.phase == 'scratch':
            assert len(new_params) > 0, "No new params to pretrain!"
            param_groups = [{'params': ft_params, 'lr': self.args.new_params_lr_promoting}]
            param_groups += [{'params': new_params, 'lr': self.args.new_params_lr_promoting}]
        elif self.args.phase == 'normal':
            param_groups = [{'params': ft_params, 'lr': self.args.ft_lr_promoting}]
            # Some model may not have new params
            if len(new_params) > 0:
                param_groups += [{'params': new_params, 'lr': self.args.new_params_lr_promoting}]
        elif self.args.phase == 'finetune':
            param_groups = [{'params': ft_params, 'lr': self.args.ft_lr_promoting}]
            param_groups += [{'params': new_params, 'lr': self.args.ft_lr_promoting}]
        else:
            param_groups = [{'params': ft_params, 'lr': 0.0}]
            param_groups += [{'params': new_params, 'lr': self.args.ft_lr_promoting}]

        self.optimizer_promoting = create_optimizer(param_groups, self.args)
        recursive_to_device(self.optimizer_promoting.state_dict(), self.device)
        return self.optimizer_promoting

    def get_ft_and_new_params(self):
        """cft: Clustering and Fine Tuning"""
        ft_modules, new_modules = self.get_ft_and_new_modules()
        ft_params = list(chain.from_iterable([list(m.parameters()) for m in ft_modules]))
        new_params = list(chain.from_iterable([list(m.parameters()) for m in new_modules]))
        return ft_params, new_params

    def get_ft_and_new_modules(self):
        ft_modules = [self.model.module.backbone]
        new_modules = [self.model.module.em_list]
        if hasattr(self.model.module, 'cls_list'):
            new_modules += [self.model.module.cls_list]
        return ft_modules, new_modules

    def create_lr_scheduler(self):
        if self.args.phase == 'normal':
            self.args.lr_decay_steps = [self.args.num_train_loader * it * self.args.epochs for it in self.args.lr_decay_iters]
            self.lr_scheduler = MultiStepLR(self.optimizer, self.args.lr_decay_steps)
        else:
            self.lr_scheduler = StepLR(self.optimizer, 10)
        return self.lr_scheduler

    def create_lr_scheduler_promoting(self, num_train_loader):
        if self.args.phase == 'normal':
            self.args.lr_decay_steps_iters = [num_train_loader * ep for ep in self.args.lr_decay_epochs]
            self.lr_scheduler_promoting = MultiStepLR(self.optimizer_promoting, self.args.lr_decay_steps_iters)
        else:
            self.lr_scheduler_promoting = StepLR(self.optimizer_promoting, 10)
        return self.lr_scheduler_promoting

    def create_loss_funcs(self):
        self.loss_funcs = IDLoss(self.args)

    def create_criterion(self):
        self.criterion = OrderedDict()
        if self.args.idloss_use:
            self.criterion[self.args.idloss_name] = IDLoss(self.args)
        if self.args.triloss_use:
            if self.args.tri_sampler_type == 'CTL':
                self.criterion[self.args.triloss_name] = CTLTripletLoss(self.args, tri_sampler_type=self.args.tri_sampler_type)
            if self.args.tri_sampler_type == 'RTL':
                self.criterion[self.args.triloss_name] = RTLTripletLoss(self.args, tri_sampler_type=self.args.tri_sampler_type)
            if self.args.tri_sampler_type == 'CTL_RTL':
                self.criterion[self.args.triloss_name + '_RTL'] = RTLTripletLoss(self.args, name=self.args.triloss_name + '_RTL', tri_sampler_type=self.args.tri_sampler_type)
                self.criterion[self.args.triloss_name + '_CTL'] = CTLTripletLoss(self.args, name=self.args.triloss_name + '_CTL', tri_sampler_type=self.args.tri_sampler_type)

        return self.criterion

    # NOTE: To save GPU memory, our multi-domain training requires
    # [1st batch: source-domain forward and backward]-
    # [2nd batch: cross-domain forward and backward]-
    # [update model]
    # So the following three-step framework is not strictly followed.
    #     pred = self.train_forward(batch)
    #     loss = self.criterion(targets, pred)
    #     loss.backward()

    def _parse_data(self, inputs, tri_loss_type='CTL'):
        if self.args.triloss_use and tri_loss_type == 'RTL':
            a_imgs, p_imgs, n_imgs, p_n_index, pids = inputs
            a_inputs, p_inputs, n_inputs = \
                a_imgs.cuda(), p_imgs.cuda(), n_imgs.cuda()
            p_n_index = [p.cuda() for p in p_n_index]
            targets = torch.cat(pids, 0).cuda()
            return [a_inputs, p_inputs, n_inputs, p_n_index], targets
        else:
            imgs, _, pids, _ = inputs
            inputs = imgs.cuda()
            targets = pids.cuda()
            return inputs, targets

    def _forward(self, inputs, targets=None, cls_list=None, tri_loss_type='CTL'):
        if tri_loss_type == 'CTL':
            return self._CTL_forward(inputs, targets=targets, cls_list=cls_list)
        elif tri_loss_type == 'RTL':
            return self._RTL_forward(inputs, targets=targets, cls_list=cls_list)
        elif tri_loss_type == 'IDL':
            return self._IDL_forward(inputs, targets=targets, cls_list=cls_list)
        else:
            return self._IDL_forward(inputs, targets=targets, cls_list=cls_list)
        
    def _IDL_forward(self, inputs, targets=None, cls_list=None):
        loss = 0
        prec = 0
        out_dict = self.model.forward(inputs, forward_type=self.args.forward_type)  # out_dict['feat_list', 'logits_list']
        if self.args.idloss_use and cls_list is not None:
            self.model.module.cls_list = copy.deepcopy(cls_list)
            self.cls_list = copy.deepcopy(cls_list)
            result = self.criterion[self.args.idloss_name](targets, out_dict)  # id_loss(targets, out_dict['logits_list']) {'loss': loss, 'prec':prec}
            # Scale by loss weight
            loss += result['loss'] * self.args.idloss_weight
            prec += result['prec']
        else:
            assert cls_list is not None, "There is no classifier layer."
            assert self.args.idloss_use, "Please open the args.idloss_use."

        return out_dict, loss, prec

    def _CTL_forward(self, inputs, targets, cls_list=None):
        loss = 0
        prec = 0
        out_dict = self.model.forward(inputs, forward_type=self.args.forward_type)  # out_dict['feat_list', 'logits_list']

        if self.args.triloss_use:
            if self.args.tri_sampler_type == 'CTL_RTL' and cls_list is None:
                tri_result = self.criterion[self.args.triloss_name + '_CTL'](targets, out_dict)  # id_loss(targets, out_dict['logits_list']) {'loss': loss, 'prec':prec
                # Scale by loss weight
                loss += tri_result['loss'] * self.args.triloss_weight * self.args.triloss_rho_ctl
            else:
                tri_result = self.criterion[self.args.triloss_name](targets, out_dict)
                # Scale by loss weight
                loss += tri_result['loss'] * self.args.triloss_weight

        return out_dict, loss, 0

    def _RTL_forward(self, inputs, targets=None, cls_list=None):
        assert self.args.triloss_use, "There is no loss functions to optimize, you should open the triloss_use"

        loss = 0

        a_out_dict = self.model.forward(inputs[0], forward_type=self.args.forward_type)
        p_out_dict = self.model.forward(inputs[1], forward_type=self.args.forward_type)
        n_out_dict = self.model.forward(inputs[2], forward_type=self.args.forward_type)
        p_n_index = inputs[3]
        if self.args.tri_sampler_type == 'CTL_RTL':
            tri_result = self.criterion[self.args.triloss_name + '_RTL'](a_out_dict, p_out_dict, n_out_dict, p_n_index=p_n_index)
            loss += tri_result['loss'] * self.args.triloss_weight * self.args.triloss_rho_rtl
        else:
            tri_result = self.criterion[self.args.triloss_name](a_out_dict, p_out_dict, n_out_dict, p_n_index=p_n_index)
            loss += tri_result['loss'] * self.args.triloss_weight

        out_dict = [a_out_dict, p_out_dict, n_out_dict]
        return out_dict, loss, 0
