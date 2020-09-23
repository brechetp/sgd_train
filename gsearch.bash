#for arg in 'softmax' 'no-softmax'; do srun -p cuda -x cuda01,cuda02 python train_mnist.py  --vary_name softmax --width 1000 --gd_mode 'full' "--$arg" & done
dir='slurm/scripts'
template='template.sbatch'
dataset=$1
[ -z $max_run ] && max_run=2;
incr=1
sbname=nl-$dataset
fname="$dir/$sbname.sbatch"
cp "$dir/$template" $fname
exit_code=0
((incr++))
c=0.36
ns=square
date=`date +%y%m%d`  # format yymmdd
root=results/$dataset/$date

sed -i "s/^\(#SBATCH -J\) test_slurm/\1 $sbname/" $fname

#for shuffle in `seq 0 0.1 1`;


for nl in `seq 7 10 | shuf`; 
do
name=ns-$ns/c-$c/nl-$nl/
echo "#srun python train_mnist.py -o $root --name $name --dataset $dataset --nlayers $nl --coefficient $c --net_shape $ns" --nepoch 1000 >> $fname; 
model=$root/$name/checkpoint-r1.pth
echo "#srun python train_lin.py --model $model --keep_ratio 0.5 --name $name" >> $fname; 
done;
#for kr in `seq 0.1 0.1 0.5` 
#for ns in square
#do
#    name=kr-$kr
#    echo "#srun python train_lin.py --model $model --keep_ratio $kr --name $name" >> $fname; 
#done;

nexp=`grep srun $fname  | wc -l`  # the number of experiments in the file
total=`wc -l < $fname`  # the total number of lines


let beg=$total-$nexp+1
let i=$beg

let blocks=($nexp - 1)/$max_run+1

for bcnt in `seq 1 $blocks`; do
    sed -i "s/^\(#SBATCH -J\) .*$/\1 $sbname-$bcnt/" $fname
    sed -i "$i,`expr $i + $max_run - 1`s/^#*//" $fname
    sbatch $fname
    sed -i "$i,`expr $i + $max_run - 1`s/^/#/" $fname
    let i=$i+$max_run
done;

