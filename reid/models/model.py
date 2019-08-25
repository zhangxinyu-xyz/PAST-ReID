from __future__ import print_function
import torch
import torch.nn as nn
from .base_model import BaseModel
from .backbone import create_backbone
from .pcb_pool import PCBPool, PCBPool_nine
from ..utils.model import create_embedding


class Model(BaseModel):
    def __init__(self, args, cls_params=None):
        super(Model, self).__init__()
        self.args = args
        self.cls_params = cls_params
        if self.cls_params is not None:
            self.num_classes = cls_params.size(0)
        else:
            self.num_classes = self.args.num_classes
        self.backbone = create_backbone(args.model_name, args)
        self.pool = eval('{}(args)'.format(args.pool_type))
        self.create_em_list()

        print('Model Name:\n{}_{}'.format(self.args.model_name, self.args.pool_type))

    def create_em_list(self):
        args = self.args
        if self.args.pool_type == 'PCBPool_nine':
            self.em_list = nn.ModuleList([create_embedding(in_dim=self.backbone.out_c, out_dim=args.embedding_dim,
                                                           local_conv=self.args.local_conv).cuda() for _ in range(args.num_parts+3)])
        else:
            self.em_list = nn.ModuleList([create_embedding(in_dim=self.backbone.out_c, out_dim=args.embedding_dim,
                                                           local_conv=self.args.local_conv).cuda() for _ in range(args.num_parts)])

    def set_train_mode(self, fix_ft_layers=False):
        self.train()
        if fix_ft_layers:
            for m in self.get_ft_and_new_modules()[0]:
                m.eval()

    def set_test_mode(self):
        self.eval()

    def backbone_forward(self, in_dict):
        return self.backbone(in_dict['im'])

    def reid_forward(self, in_dict):
        # print('=====> in_dict.keys() entering reid_forward():', in_dict.keys())
        pool_out_dict = self.pool(in_dict)
        feat_list = [em(f) for em, f in zip(self.em_list, pool_out_dict['feat_list'])]
        feat_list = [f.view(f.size(0), -1) for f in feat_list]
        out_dict = {
            'feat_list': feat_list,
        }
        return out_dict

    def forward(self, inputs, forward_type='reid'):
        in_dict = {}
        in_dict['im'] = inputs
        in_dict['feat'] = self.backbone_forward(in_dict)
        if forward_type == 'reid':
            out_dict = self.reid_forward(in_dict)
        else:
            raise ValueError('Error forward_type {}'.format(forward_type))
        return out_dict
