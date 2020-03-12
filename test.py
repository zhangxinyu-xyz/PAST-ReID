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
    parser.add_argument('--batch_size', type=int, default=16)
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
    parser.add_argument('--load_optimizer', action='store_true',
                        help='when continue training, whether load optimizer and scheduler or not.')
    parser.add_argument('--save_freq', type=int, default=20, help='how many epochs to save checkpoint')
    
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
    
    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, help='the parameter of SGD')
    parser.add_argument('--nesterov', type=bool, default=False, help='the parameter of SGD')
    parser.add_argument('--ft_lr', type=float, default=0.0001,
                        help="learning rate of finetune parameters")
    parser.add_argument('--new_params_lr', type=float, default=0.0002, help="learning rate of new parameters")
    parser.add_argument('--lr_decay_epochs', nargs='*', type=int, default=25, help="epoch for lr descend")
    parser.add_argument('--num_train_loader', type=bool, default=False, help="epoch for lr descend * len(train_loader)")
    parser.add_argument('--start_epochs', type=int, default=0, help="start_epochs")
    parser.add_argument('--epochs', type=int, default=60, help="max_epochs")
    
    parser.add_argument('--ft_lr_promoting', type=float, default=0.00005,
                        help="learning rate of finetune parameters for supervision")
    parser.add_argument('--new_params_lr_promoting', type=float, default=0.001,
                        help="learning rate of new parameters for supervision")
    parser.add_argument('--lr_decay_iters', nargs='*', type=int, default=8, help="iters for lr descend")
    parser.add_argument('--start_iters', type=int, default=0, help="start_iters")
    parser.add_argument('--iters', type=int, default=10, help="max_iters")
    
    parser.add_argument('--dbscan_type', type=str, default='hdbscan', choices=['dbscan', 'hdbscan'])
    parser.add_argument('--dbscan_use', type=bool, default=True, help='whether use dbscan or not')
    parser.add_argument('--dbscan_iter', type=int, default=20, help="dbscan_iter epochs to calculate once dbscan")
    parser.add_argument('--dbscan_minsample', type=int, default=10, help="dbscan_minsample")
    parser.add_argument('--start_dbscan', type=int, default=-1, help="start_epochs")
    
    # loss
    parser.add_argument('--idloss_use', action='store_true', help='whether use ID loss or not')
    parser.add_argument('--idloss_weight', type=float, default=0.1, help='ID loss weight')
    parser.add_argument('--idloss_name', type=str, default='idL', help='ID loss name')
    parser.add_argument('--idloss_iter', type=int, default=1, help="idloss_iter epochs to calculate once idloss")
    
    parser.add_argument('--triloss_use', action='store_true', help='whether use Triplet loss or not')
    parser.add_argument('--triloss_weight', type=float, default=0.1, help='Triplet loss weight')
    parser.add_argument('--triloss_rho_ctl', type=float, default=1.0, help='Triplet loss weight')
    parser.add_argument('--triloss_rho_rtl', type=float, default=1.0, help='Triplet loss weight')
    parser.add_argument('--triloss_name', type=str, default='triL', help='Triplet loss name')
    parser.add_argument('--triloss_mean', action='store_true', help='part triplet loss sum or mean')
    # parser.add_argument('--triloss_online', action='store_true', help='part triplet loss sum or mean')
    parser.add_argument('--triloss_part', type=int, default=9, help='Triplet loss part sum or mean')
    
    parser.add_argument('--tri_sampler_type', type=str, default='CTL_RTL', help='Triplet sampler type',
                        choices=['CTL', 'RTL', 'CTL_RTL'])
    parser.add_argument('--k_nearest', type=int, default=20, help='k_nearest for tri_sampler_type==softmargintriplet')
    parser.add_argument('--hard_type', type=str, default='tri_hard',
                        help='batch triplets selection when tri_sampler_type==RandomIdentitySampler')
    parser.add_argument('--margin', type=float, default=0.3, help='Triplet loss margin')
    parser.add_argument('--num_instances', type=int, default=4,
                        help="number of instances per identity (if use triplet loss)")
    parser.add_argument('--norm_by_num_of_effective_triplets', type=bool, default=False,
                        help='Triplet loss norm_by_num_of_effective_triplets')
    
    # rerank misc parameters
    parser.add_argument('--rerank', action='store_true', help="rerank or not")
    parser.add_argument('--dist_epoch', nargs='*', type=int, default=0,
                        help="epoch for rerank distmat, the sequence is the same with test_names")
    parser.add_argument('--k1', type=int, default=20, help="rerank k1")
    parser.add_argument('--k2', type=int, default=6, help="rerank k2")
    parser.add_argument('--lambda_value', type=float, default=0.3, help='rerank lambda_value')
    parser.add_argument('--rerank_eval', action='store_true', help="after rerank, whether evaluate or not")
    parser.add_argument('--rerank_dist_file', type=str, default='', metavar='PATH',
                        help='initial rerank distmat file path')
    parser.add_argument('--rho', type=float, default=1.6e-3,
                        help="rho percentage, default: 1.6e-3")
    parser.add_argument('--init_t_t_f', type=str, default='', metavar='PATH',
                        help='initial target train features file path')
    main(parser.parse_args())
