#!/bin/bash

loss=sampled_povey
data=data/ami
batch_size=64
dropout=0.4
noise_ratio=512
lr=0.001
concat=true
sample_with_replacement=false
sample_with_grouping=false
emsize=200
epochs=200
nlayers=2
log_interval=20
norm_term=1.0
normalize=0
trick=0

. parse_options.sh

if $concat; then
  concat="--concat"
else
  concat=""
fi

if $sample_with_replacement; then
  sample_with_replacement="--sample-with-replacement"
else
  sample_with_replacement=""
fi

if $sample_with_grouping; then
  sample_with_grouping="--sample-with-grouping"
else
  sample_with_grouping=""
fi

export CUDA_VISIBLE_DEVICES=`free-gpu`

dir=ami_${loss}_${batch_size}_${dropout}_${noise_ratio}_${lr}_${nlayers}_${norm_term}_${normalize}_${trick}
mkdir -p saved_model/$dir
mkdir -p log/$dir

python3 -u main.py $concat $sample_with_replacement $sample_with_grouping --data $data --norm-term $norm_term --log-interval $log_interval --nlayers $nlayers --epochs $epochs --emsize $emsize --lr $lr --batch-size $batch_size --cuda --loss $loss --train --noise-ratio $noise_ratio --save $dir/model 2>&1 | tee saved_model/$dir/log.train
best_epoch=`grep "valid ppl" saved_model/$dir/log.train | awk '{print NR, $3}' | sed "s=,==g" | sort -k2n | head -n1 | awk '{print $1}'`
best_loss=`grep "valid ppl" saved_model/$dir/log.train | awk '{print NR, $3}' | sed "s=,==g" | sort -k2n | head -n1 | awk '{print $2}'`

cp $data/vocab.pkl saved_model/$dir
echo $best_loss > saved_model/$dir/best_loss

rm -f $PWD/saved_model/$dir/best.mdl
ln -s $PWD/saved_model/$dir/model.epoch_$best_epoch $PWD/saved_model/$dir/best.mdl
