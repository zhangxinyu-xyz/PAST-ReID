from __future__ import print_function
from copy import deepcopy
from .trainer import ReIDTrainer
from ..models.model import Model
from torch import nn


class PCBTrainer(ReIDTrainer):

    def set_model_to_train_mode(self):
        self.model.module.set_train_mode(fix_ft_layers=self.args.phase == 'fix_finetune_layers')

    def set_model_to_test_mode(self):
        self.model.module.set_test_mode()

    def create_model(self):
        self.model = Model(deepcopy(self.args), self.cls_params)
        self.model = nn.DataParallel(self.model).cuda()

