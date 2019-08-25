#!/bin/bash
# Source market1501, Target duke. Test process.

CUDA_VISIBLE_DEVICES=0 python test.py --s_name market1501 --name duke --model_name resnet50 --batch_size 128 --embedding_dim 256 --height 384 --width 128 --local_conv --continue_training --log_dir ./results/M-D/D/ --save_dir ./results/M-D/D/ --resume_file ./best_model/M-D_best-model.pth.tar --evaluate



