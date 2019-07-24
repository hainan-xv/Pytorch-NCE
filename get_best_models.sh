#!/bin/bash


for dir in `ls saved_model`; do
  rm $PWD/saved_model/$dir/best.mdl
  best_epoch=`grep "valid ppl" saved_model/$dir/log.train | awk '{print NR, $3}' | sed "s=,==g" | sort -k2n | head -n1 | awk '{print $1}'`

  ln -s $PWD/saved_model/$dir/model.epoch_$best_epoch $PWD/saved_model/$dir/best.mdl

done

false && for dir in `ls saved_model`; do
  best_epoch=`grep "valid ppl" saved_model/$dir/log.train | awk '{print NR, $3}' | sed "s=,==g" | sort -k2n | head -n1 | awk '{print $1}'`
  echo $dir
  grep "diagnostic" saved_model/$dir/log.train | awk -v b=$best_epoch 'NR==b'
  grep "valid ppl" saved_model/$dir/log.train | awk -v b=$best_epoch 'NR==b'
  echo 
done
