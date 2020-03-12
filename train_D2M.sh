#!/bin/bash
# Source duke, Target market1501. Using market1501 to generate triplets and finetune.

CUDA_VISIBLE_DEVICES=0,1 python train.py --s_name duke --name market1501 --model_name resnet50 --epochs 40 --embedding_dim 256 --height 384 --width 128 --lr_decay_epochs 40 --local_conv --tri_sampler_type CTL_RTL --triloss_rho_ctl 0.5 --triloss_rho_rtl 1.0 --iters 4 --lr_decay_iters 3 4 --triloss_use --idloss_use --continue_training --resume_file ./initialization/pretrained_model/source_D.pth.tar --init_t_t_f ./initialization/initial_feature/D-M_M-t-feature.mat --rerank_dist_file ./initialization/initial_distmat/D-M_M-t-rerank-distmat.mat --log_dir ./results/D-M/log/ --save_dir ./results/D-M/checkpoint/ 




