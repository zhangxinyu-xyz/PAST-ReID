from __future__ import print_function
import torch
import torch.nn as nn
from .base_model import BaseModel
from .backbone import create_backbone
from .pcb_pool import PCBPool, PCBPool_nine
from ..utils.model import create_embedding, init_classifier


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
        if hasattr(args, 'num_classes') and args.num_classes > 0 and args.idloss_use:
            self.create_cls_list()
            print('Model Structure:\n{}'.format(self.cls_list))

        print('Model Name:\n{}_{}'.format(self.args.model_name, self.args.pool_type))

    def create_em_list(self):
        args = self.args
        if self.args.pool_type == 'PCBPool_nine':
            self.em_list = nn.ModuleList([create_embedding(in_dim=self.backbone.out_c, out_dim=args.embedding_dim,
                                                           local_conv=self.args.local_conv).cuda() for _ in range(args.num_parts+3)])
        else:
            self.em_list = nn.ModuleList([create_embedding(in_dim=self.backbone.out_c, out_dim=args.embedding_dim,
                                                           local_conv=self.args.local_conv).cuda() for _ in range(args.num_parts)])

    def create_cls_list(self):
        args = self.args
        if self.args.pool_type == 'PCBPool_nine':
            self.cls_list = nn.ModuleList(
                [nn.Linear(args.embedding_dim, self.num_classes) for _ in range(args.num_parts + 3)])
        else:
            self.cls_list = nn.ModuleList(
                [nn.Linear(args.embedding_dim, self.num_classes) for _ in range(args.num_parts)])

        ori_w = self.cls_list[0].weight.view(-1).detach().numpy().copy()
        self.cls_list.apply(init_classifier)
        new_w = self.cls_list[0].weight.view(-1).detach().numpy().copy()
        import numpy as np
        if np.array_equal(ori_w, new_w):
            from ..utils.logging import array_str
            print('!!!!!! Warning: Model Weight Not Changed After Init !!!!!')
            print('Original Weight [:20]:\n\t{}'.format(array_str(ori_w[:20], fmt='{:.6f}')))
            print('New Weight [:20]:\n\t{}'.format(array_str(new_w[:20], fmt='{:.6f}')))

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
        if hasattr(self, 'cls_list'):
            logits_list = [cls.cuda()(f) for cls, f in zip(self.cls_list, feat_list)]
            out_dict['logits_list'] = logits_list
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
