#!/bin/bash

device=$1

python main.py --threshold 0.9999 --epochs 1 --kill_cnt 1 --dataset ogbl-collab_CN_2_1_0 --device $device --runs 10 --score_model NCN --model gcn --ln --lnnn --res --use_xlin --gnnlr 0.0001 --prelr 0.0001 --gnndp 0.0 --predp 0.0 --target_kl 100.0 --train_per 1 --dynamic --batch_size 32 --hidden_channels 256