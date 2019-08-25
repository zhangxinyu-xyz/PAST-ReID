from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os

import numpy as np
import sys
import torch
from torch.backends import cudnn

from reid import datasets
from reid import models
from reid.utils.data import create_test_data_loader
from reid.utils.logging import Logger
from reid.optim.pcb_trainer import PCBTrainer
from reid.eval.rerankor import Rerankor

def main(args):
    assert args.evaluate, "This version of code is only for test. We will release the training code later.\n" \
                          "If you want to use this code, please set the config with '--evaluate'"
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.log_dir, 'log_test.txt'))

    args.num_classes = 1
    
    # Create data loaders
    dataset = {}
    dataset['dataset'] = datasets.create(args.name, args.data_dir)
    dataset['train_loader'], dataset['query_loader'], dataset['gallery_loader'] \
        = create_test_data_loader(args, args.name, dataset['dataset'])
       
    if args.evaluate:
        cls_params = None
        trainer = PCBTrainer(args, cls_params=cls_params)
        evaluator = trainer.test()
        scores = {}

        scores['cmc_scores'], scores['mAP'], q_f, g_f, _ = \
            evaluator.evaluate(args.name, dataset['query_loader'], dataset['gallery_loader'],
                               dataset['dataset'].query, dataset['dataset'].gallery, isevaluate=True)
        
        print('Cross Ddomain CMC Scores')
        print('Source\t Target\t Top1\t Top5\t Top10\t MAP')
        print('{}->{}: {:6.2%} {:6.2%} {:6.2%} ({:.2%})'.format(args.s_name, args.name,
                                                                scores['cmc_scores'][0],
                                                                scores['cmc_scores'][1],
                                                                scores['cmc_scores'][2],
                                                                scores['mAP']))

        ################## whether rerank test ############
        if args.rerank:
            rerankor = Rerankor()
            rerankor.rerank(q_f, g_f,
                               savepath=os.path.join(args.save_dir, 'rerank'),
                               save=False, isevaluate=True,
                               dataset=dataset['dataset'])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PAST")
    # data
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='./data/')
    parser.add_argument('--name', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('--log_dir', type=str, metavar='PATH',
                        default='./log')
    parser.add_argument('--save_dir', type=str, metavar='PATH', default='./checkpoint',
                        help='Directory to store experiment output, including model checkpoint and tensorboard files, etc.')
    parser.add_argument('--s_name', type=str, default='market1501', help='pretrained source dataset name')
    
    # train parameters
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine_trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--phase', type=str, default='normal',
                        choices=['scratch', 'normal', 'finetune', 'fix_finetune_layers'])
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--continue_training', action='store_true',
                        help='whether continue training or not. if True, continue from the intermediate status')
    
    # test parameters
    parser.add_argument('--dist_type', type=str, default='cosine', choices=['euclidean', 'cosine'])
    parser.add_argument('--resume_file', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    
    # model
    parser.add_argument('--model_name', type=str, default='resnet50', choices=models.names())
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--num_parts', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--backbone_pretrained', type=bool, default=True)
    parser.add_argument('--backbone_pretrained_model_dir', type=str, default='initialization/pretrained_model/', metavar='PATH')
    parser.add_argument('--last_conv_stride', type=int, default=1)
    parser.add_argument('--local_conv', action='store_true', help='last_conv using linear or conv')
    parser.add_argument('--max_or_avg', type=str, default='max')
    parser.add_argument('--pool_type', type=str, default='PCBPool_nine', choices=['PCBPool', 'PCBPool_nine'])
    parser.add_argument('--forward_type', type=str, default='reid')
    
    # rerank misc parameters
    parser.add_argument('--rerank', action='store_true', help="rerank or not")
    parser.add_argument('--k1', type=int, default=20, help="rerank k1")
    parser.add_argument('--k2', type=int, default=6, help="rerank k2")
    parser.add_argument('--lambda_value', type=float, default=0.3, help='rerank lambda_value')
    parser.add_argument('--rerank_eval', action='store_true', help="after rerank, whether evaluate or not")
    parser.add_argument('--rho', type=float, default=1.6e-3,
                        help="rho percentage, default: 1.6e-3")
    main(parser.parse_args())
