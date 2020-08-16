#!/bin/bash

exit_code=0
c=1
incr=1
dataset=$1
nt=$2
date=`date +%y%m%d`  # format yymmdd
root=results/$dataset/$date



while [ $exit_code -eq 0 ]
do
    name=nt-$nt/c-$c
    srun -p cuda -x cuda01,cuda02 python train_mnist.py --dataset $dataset --net_topology $nt --coefficient $c  --name $name -o $root
    model=$root/$name/checkpoint.pth
    srun -p cuda -x cuda01,cuda02 python train_lin.py --model $model 
    exit_code=$?
let c++
done
