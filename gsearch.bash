#for arg in 'softmax' 'no-softmax'; do srun -p cuda -x cuda01,cuda02 python train_mnist.py  --vary_name softmax --width 1000 --gd_mode 'full' "--$arg" & done
[ -z $max_run ] && max_run=1;
dir='slurm/scripts'
template='template.sbatch'
dataset=cifar10
#width=500
#depth=4
sbname="bcw"
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
#echo $model
#for W in `seq 50 50 500`; do
#for el in 4; do
#for var in 1; do #od9; do #`seq 1 4`; do 
    #L=2
#for W in `seq 50 50 500`;  do
#for
#f=2
#model="results/mnist/210127/L-2/w-$W/checkpoint.pth"
#model="results/cifar10/210205/checkpoint.pth"
#model="results/cifar10/210120/vgg-11/checkpoint.pth"
#L=2
for model in $models; do
#while read model; do
   #for el in `seq 0 5`; do #`seq 2 5`; do
    #for el in `seq `; do
    #for W in `seq 1800 300 3000`; do #seq 50 50 500` `seq 600 100 1500`; do #`seq 2000 500 3000`; do
        #[[ $W -lt 200 ]] && lr="0.001" || lr="0.01"
        #lr=0.0005
        #lr=0.0005
        #echo "#srun python train_vgg.py --model vgg-11 --name vgg/var-$var" >> $fname;
    #model="results/mnist/210127/L-2/w-$W/checkpoint.pth"
    #name="fraction-$f-try-20"
    #for el in `seq 0 2 | shuf`;  do
    #for el in 0 1 2; do #1 2; do
        #name="f2-lr-5e-4"
        #(( $el == 0 )) && lr=0.001 || lr=0.005
        #lr=0.0005
        #name=ns-$ns/c-$c/nl-$nl/
        #echo "#srun python train_mnist.py --dataset $dataset -L $L --vary_name width depth --width $width" >> $fname; 
        #model=$root/$name/checkpoint-r1.pth
        #echo "#srun python annex_vgg.py --model $model --fraction $f --entry_layer $L --name $name --draws 20 -lrm manual -lr 0.002 -lrs 0  --nepochs 1000 ">> $fname; 
        #echo "#srun python train_mnist.py --vary_name lr_mode depth width --width $W --depth $L --dataset $dataset" --lr_mode hessian  >> $fname
        #fn_chkpt=`dirname $model`/ds-f2_optim-mult/eval_copy.pth
        #if [[ -f $fn_chkpt ]]; then
            #echo "#srun python eval_copy.py --checkpoint $fn_chkpt"  >> $fname
        #else
        echo "#srun python eval_copy.py --model $model --optim_mult --name ds-f2 --steps 200"  >> $fname
        #fi
        #echo "#srun python train_fcn.py -oroot $oroot --name L-$L/W-$W/var-$var --width $W --depth $L --dataset $dataset -lr $lr -lrm manual"  >> $fname
        #echo "#srun python train_fcn.py --name var-$var --width $width --depth $depth --dataset $dataset -lr 0.005 -lrm manual"  >> $fname
        #echo "#srun python check_seq.py --model $model --learning_rate $lr --entry_layer $el --fraction $f --min_epochs 100 --name $name -lrs 0 --nepoch 1000" >> $fname
        #echo "#srun python check_seq.py --checkpoint $model/fraction-2/checkpoint_entry_2.pth "  >> $fname
        #echo "#srun python check_seq.py --model $model --learning_rate 0.0005 --entry_layer $el --name f2 --min_epochs 100 --fraction 2 -lrm manual -lrs 0 "  >> $fname
    #done < $model_list;
done;
#done;
#done;
#for kr in `seq 0.1 0.1 0.5` 
#for ns in square
#do
cat $fname
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
    sed -i "$i,`expr $i + $max_run - 1`s/^#*//" $fname  # uncomment the range
    sbatch $fname
    sleep 1
    sed -i "$i,`expr $i + $max_run - 1`s/^/#/" $fname  # comment the range
    let i=$i+$max_run
done;
sed -i "`expr $i - $max_run + 1`,${i}s/^#*//" $fname  # uncomment the range

