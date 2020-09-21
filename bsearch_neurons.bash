#!/bin/bash

exit_code=0
tol=0.01
supb=1
infb=0
dataset=cifar10
ntry=$2
date=`date +%y%m%d`  # format yymmdd
root=results/$dataset/$date
net_shape=$1
#model=$1
#model=$root/$name/checkpoint.pth

while (( $(echo "$supb-$infb > $tol" | bc -l)));
do
    c=`echo $(echo "($supb+$infb)/2" | bc -l) | awk '{printf "%f", $0}'` # the keep ratio
    #srun -p cuda -x cuda01,cuda02 python train_mnist.py --dataset $dataset --net_topology $nt --coefficient $c  --name $name -o $root
    name="ns-$net_shape/c-$c"
    fname=$root/${name%/c-*}/bounds.txt
    mkdir -p $root/$name
    echo "bounds: $infb $supb, c: $c" >> $fname
    srun -p cuda -x cuda01,cuda02 python train_mnist.py --dataset cifar10 --coefficient $c --net_shape $net_shape --name $name
    exit_code=$?
    if (( $exit_code == 0 )); then # the program terminated with separated data
        echo "success" >> $fname
        supb=$c  # reduce the uppor bound
    else  # failure
        echo "failure" >> $fname
        infb=$c
    fi
    #echo "new bounds: $infb $supb"
done

echo "bounds: $infb $supb" >> $fname
echo "stop" >> $fname
