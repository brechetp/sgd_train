#for arg in 'softmax' 'no-softmax'; do srun -p cuda -x cuda01,cuda02 python train_mnist.py  --vary_name softmax --width 1000 --gd_mode 'full' "--$arg" & done
[ -z $max_run ] && max_run=5;
dir='slurm/scripts'
template='template.sbatch'
dataset=cifar10
#width=500
#depth=4
sbname="c-v"
#width=$2
models=$@
#oroot=$1
oroot="results/cifar10/210510/"
f=2
name=fraction-4
#incr=1
#sbname="a-c5-2"
fname="$dir/$sbname.sbatch"
cp "$dir/$template" $fname
exit_code=0
#model_list=$1
#((incr++))
#c=0.36
#ns=square
#date=`date +%y%m%d`  # format yymmdd
#root=results/$dataset/$date

sed -i "s%^\(#SBATCH -J\) test_slurm%\1 $sbname%" $fname
sed -i "s%^\(#SBATCH -o .*/\).*%\1/$sbname.out%" $fname
sed -i "s%^\(#SBATCH -e .*/\).*%\1/$sbname.err%" $fname

#for shuffle in `seq 0 0.1 1`; 

#for model in $models; do
#for W in `seq 50 50 500`; do
#for W in `seq 50 50 500`; do
#for el in 4; do
for var in 9; do #`seq 1 4`; do 
    #L=2
#for W in `seq 50 50 500`;  do
#for
f=2
#model="results/mnist/210127/L-2/w-$W/checkpoint.pth"
#model="results/cifar10/210205/checkpoint.pth"
#model="results/cifar10/210120/vgg-11/checkpoint.pth"
#for model in $models; do
#while read model; do
    #for el in `seq 0 5`; do #`seq 2 5`; do
    #for el in `seq `; do
    #for W in `seq 50 50 500` `seq 600 100 1500`; do #`seq 2000 500 3000`; do
        #[[ $W -lt 200 ]] && lr="0.001" || lr="0.01"
        #lr=0.0005
        echo "#srun python train_vgg.py --model vgg-11 --name vgg/var-$var" >> $fname;
    #model="results/mnist/210127/L-2/w-$W/checkpoint.pth"
    #name="fraction-$f-try-20"
    #for L in `seq 0 10 | shuf`;  do
        #name="f2"
        #name=ns-$ns/c-$c/nl-$nl/
        #echo "#srun python train_mnist.py --dataset $dataset -L $L --vary_name width depth --width $width" >> $fname; 
        #model=$root/$name/checkpoint-r1.pth
        #echo "#srun python annex_vgg.py --model $model --fraction $f --entry_layer $L --name $name --draws 20 -lrm manual -lr 0.002 -lrs 0  --nepochs 1000 ">> $fname; 
        #echo "#srun python train_mnist.py --vary_name lr_mode depth width --width $W --depth $L --dataset $dataset" --lr_mode hessian  >> $fname
        #echo "#srun python eval_copy.py --model $model --optim_mult --name ds-f2 --steps 200"  >> $fname
        #echo "#srun python train_fcn.py -oroot $oroot --name L-$L/W-$W/var-$var --width $W --depth $L --dataset $dataset -lr $lr -lrm manual"  >> $fname
        #echo "#srun python train_fcn.py --name var-$var --width $width --depth $depth --dataset $dataset -lr 0.005 -lrm manual"  >> $fname
        #echo "#srun python check_seq.py --model $model --learning_rate 0.05 --entry_layer $el --name fraction-2 --fraction 2 --lr_step -1 -lrg 0.5 --nepoch 1000" >> $fname
        #echo "#srun python check_seq.py --checkpoint $model/fraction-2/checkpoint_entry_2.pth "  >> $fname
        #echo "#srun python check_seq.py --model $model --learning_rate 0.0001 --entry_layer $el --name f-2-me-100 --min_epochs 100 --fraction 2 -lrm manual -lrs 0 --gd_mode full"  >> $fname
        #echo "#srun python eval_copy.py --model $model --optim_mult --steps 200 --name ds-200-f2-min"  >> $fname
    #done < $model_list;
#done;
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
    sleep 1
    sed -i "$i,`expr $i + $max_run - 1`s/^/#/" $fname
    let i=$i+$max_run
done;

