#!/bin/bash

exit_code=0
tol=0.01
supb=1
infb=0
ntry=$2
date=`date +%y%m%d`  # format yymmdd
model=$1
#model=$root/$name/checkpoint.pth

while (( $(echo "$supb-$infb > $tol" | bc -l)));
do
    kr=`echo $(echo "($supb+$infb)/2" | bc -l) | awk '{printf "%f", $0}'` # the keep ratio
    name=ntry-$ntry-kr-$kr
    fname=${model%/*.pth}/bounds.txt
    echo "bounds: $infb $supb, kr: $kr" >> $fname
    #srun -p cuda -x cuda01,cuda02 python train_mnist.py --dataset $dataset --net_topology $nt --coefficient $c  --name $name -o $root
    srun -p cuda -x cuda01,cuda02 python train_lin.py --model $model  --keep_ratio $kr --name $name --ntry $ntry
    exit_code=$?
    if (( $exit_code == 0 )); then # failure (i.e. the program terminated)
        echo "failure"
        infb=$kr
    else  # sucess, have to decrease the ratio
        echo "success"
        supb=$kr
    fi
done
echo "bounds: $infb $supb" >> $fname
echo "stop" >> $fname

