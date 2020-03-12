#!/bin/bash
# Source market1501, Target duke. Using duke to generate triplets and finetune.

CUDA_VISIBLE_DEVICES=0,1 python train.py --s_name market1501 --name duke --model_name resnet50 --epochs 40 --embedding_dim 256 --height 384 --width 128 --lr_decay_epochs 40 --local_conv --tri_sampler_type CTL_RTL --triloss_rho_ctl 0.5 --triloss_rho_rtl 1.0 --iters 4 --lr_decay_iters 3 4 --triloss_use --idloss_use --continue_training --resume_file ./initialization/pretrained_model/source_M.pth.tar --init_t_t_f ./initialization/initial_feature/M-D_D-t-feature.mat --rerank_dist_file ./initialization/initial_distmat/M-D_D-t-rerank-distmat.mat --log_dir ./results/M-D/log/ --save_dir ./results/M-D/checkpoint/ 




