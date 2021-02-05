#!/bin/bash
model=$1
dir='slurm/scripts'
template='template.sbatch'
sbname=${model#results/}
sbname=${sbname%/checkpoint.pth}
sbname=${sbname//\//-}
batch_file="$dir/$sbname.sbatch"
cp "$dir/$template" $batch_file

exit_code=0
tol=5  # the tolerance 
supb=$2
infb=0
ntry=10
date=`date +%y%m%d`  # format yymmdd
#model=$root/$name/checkpoint.pth
L=${sbname%.sbatch}
L=${L##*-}
bname_root="${sbname:0:1}L$L"

while (( $(echo "$supb-$infb > $tol" | bc -l)));
do
    #kr=`echo $(echo "0.5+($supb+$infb)/2" | bc -l) | awk '{printf "%d", $0}'` # the keep ratio
    # the removed neurons
    rs=`echo $(echo "0.5+($supb+$infb)/2" | bc -l) | awk '{printf "%d", $0}'` # the removed neurons
    name=rs-$rs
    bname="${bname_root}R${rs}"
    bounds_file=${model%/*.pth}/bounds.txt
    echo "bounds: $infb $supb, r: $rs" >> $bounds_file
    #srun -p cuda -x cuda01,cuda02 python train_mnist.py --dataset $dataset --net_topology $nt --coefficient $c  --name $name -o $root
    cp "$dir/$template" $batch_file
    echo "python train_simple_classifier.py --model $model --remove $rs --name $name --ntry $ntry" >> $batch_file
    sed -i "s/^#SBATCH -J test_slurm/#SBATCH -J $bname/" $batch_file
    sbatch -W $batch_file
    exit_code=$?
    if (( $exit_code == 0 )); then # success return code
        echo "success" >> $bounds_file
        infb=$rs  # can lower the removed
    else  # sucess, have to decrease the ratio
        echo "failure" >> $bounds_file
        supb=$rs
    fi
done
echo "bounds: $infb $supb" >> $bounds_file
echo "stop" >> $bounds_file
#
