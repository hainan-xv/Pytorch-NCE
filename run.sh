#!/bin/bash

loss=povey
batch_size=64
dropout=0.2
noise_ratio=512

. parse_options.sh

export CUDA_VISIBLE_DEVICES=`free-gpu`

dir=${loss}_${batch_size}_${dropout}_${noise_ratio}
mkdir -p saved_model/$dir
mkdir -p log/$dir

python3 -u main.py --concat --batch-size $batch_size --lr 0.1 --cuda --loss $loss --train --noise-ratio $noise_ratio --save $dir/model 2>&1 | tee log.train
