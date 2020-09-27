#!/bin/bash

exit_code=0
tol=10
supb=3979
infb=3940
ntry=10
date=`date +%y%m%d`  # format yymmdd
model=$1
#model=$root/$name/checkpoint.pth

while (( $(echo "$supb-$infb > $tol" | bc -l)));
do
    #kr=`echo $(echo "0.5+($supb+$infb)/2" | bc -l) | awk '{printf "%d", $0}'` # the keep ratio
    # the removed neurons
    rs=`echo $(echo "0.5+($supb+$infb)/2" | bc -l) | awk '{printf "%d", $0}'` # the keep ratio
    name=ntry-$ntry/rs-$rs
    fname=${model%/*.pth}/bounds.txt
    echo "bounds: $infb $supb, r: $rs" >> $fname
    #srun -p cuda -x cuda01,cuda02 python train_mnist.py --dataset $dataset --net_topology $nt --coefficient $c  --name $name -o $root
    srun -p cuda -x cuda01,cuda02 python deep_lin.py --model $model --Rs $rs --name $name --ntry $ntry
    exit_code=$?
    if (( $exit_code == 0 )); then # success return code
        echo "success" >> $fname
        infb=$rs
    else  # sucess, have to decrease the ratio
        echo "failure" >> $fname
        supb=$rs
    fi
done
echo "bounds: $infb $supb" >> $fname
echo "stop" >> $fname

