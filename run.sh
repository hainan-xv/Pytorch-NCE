#!/bin/bash

loss=povey
batch_size=64
dropout=0.0
noise_ratio=512
lr=0.001
concat=true
emsize=200
epochs=40
nlayers=2
log_interval=20

. parse_options.sh

if $concat; then
  concat="--concat"
else
  concat=""
fi

export CUDA_VISIBLE_DEVICES=`free-gpu`

dir=${loss}_${batch_size}_${dropout}_${noise_ratio}_${lr}_${nlayers}
mkdir -p saved_model/$dir
mkdir -p log/$dir

python3 -u main.py $concat --log-interval $log_interval --nlayers $nlayers --epochs $epochs --emsize $emsize --lr $lr --batch-size $batch_size --cuda --loss $loss --train --noise-ratio $noise_ratio --save $dir/model 2>&1 | tee log.train
