#for arg in 'softmax' 'no-softmax'; do srun -p cuda -x cuda01,cuda02 python train_mnist.py  --vary_name softmax --width 1000 --gd_mode 'full' "--$arg" & done
dir='slurm/scripts'
template='template.sbatch'
dataset=mnist
#width=$2
model=$1
f=4
name=fraction-4
[ -z $max_run ] && max_run=1;
#incr=1
sbname="a-m-100"
fname="$dir/$sbname.sbatch"
cp "$dir/$template" $fname
exit_code=0
#((incr++))
#c=0.36
#ns=square
#date=`date +%y%m%d`  # format yymmdd
#root=results/$dataset/$date

sed -i "s%^\(#SBATCH -J\) test_slurm%\1 $sbname%" $fname
sed -i "s%^\(#SBATCH -o .*/\).*%\1/$sbname.out%" $fname
sed -i "s%^\(#SBATCH -e .*/\).*%\1/$sbname.err%" $fname

#for shuffle in `seq 0 0.1 1`;


#for L in 0 1;do 
for W in `seq 100 100 500`; do
    L=2
#for W in `seq 50 50 300`;  do
#for
model="results/mnist/210127/L-2/w-$W/checkpoint.pth"
#model="results/cifar10/210205/checkpoint.pth"
#model="results/cifar10/210120/vgg-11/checkpoint.pth"
    for el in `seq 0 2`; do
    #for W in 100 700 1300; do
    #model="results/mnist/210127/L-2/w-$W/checkpoint.pth"
    #name="fraction-$f-try-20"
    #for L in `seq 0 10 | shuf`; 
        #name=ns-$ns/c-$c/nl-$nl/
        #echo "#srun python train_mnist.py --dataset $dataset -L $L --vary_name width depth --width $width" >> $fname; 
        #model=$root/$name/checkpoint-r1.pth
        #echo "#srun python annex_vgg.py --model $model --fraction $f --entry_layer $L --name $name --draws 20 -lrm manual -lr 0.1" -lrs 30 --nepochs 1000 >> $fname; 
        #echo "#srun python train_mnist.py --vary_name lr_mode depth width --width $W --depth $L --dataset $dataset" --lr_mode hessian  >> $fname
        #echo "#srun python eval_copy.py --model $model --optim_mult --name scalar-brent --steps 20"  >> $fname
        #echo "#srun python train_fcn.py --vary_name depth width --width $W --depth $L --dataset $dataset -oroot results/mnist/210127/ -lrm hessian "  >> $fname
        echo "#srun python check_seq.py --model $model --learning_rate 0.05 --entry_layer $el --name fraction-2 --fraction 2 --lr_step 30"  >> $fname
        #echo "#srun python check_seq.py --model $model --learning_rate 0.01 --entry_layer $el --name fraction-2 --fraction 2 --lr_step 30"  >> $fname
        #echo "#srun python eval_copy.py --model $model --optim_mult --steps 200 --name ds-200-f2"  >> $fname
    done;
done;
#done;
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
    sed -i "s%^\(#SBATCH -o .*/\).*%\1/$sbname-$bcnt.out%" $fname
    sed -i "s%^\(#SBATCH -e .*/\).*%\1/$sbname-$bcnt.err%" $fname
    sed -i "$i,`expr $i + $max_run - 1`s/^#*//" $fname
    sbatch $fname
    sed -i "$i,`expr $i + $max_run - 1`s/^/#/" $fname
    let i=$i+$max_run
done;

