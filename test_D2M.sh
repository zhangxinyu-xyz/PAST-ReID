#!/bin/bash
# Source duke, Target market1501. Test process.

CUDA_VISIBLE_DEVICES=0 python test.py --s_name duke --name market1501 --model_name resnet50 --batch_size 128 --embedding_dim 256 --height 384 --width 128 --local_conv --tri_sampler_type CTL_RTL --triloss_use --idloss_use --continue_training --log_dir ./results/D-M/M/ --save_dir ./results/D-M/M/ --resume_file ./best_model/D-M_best-model.pth.tar --evaluate


