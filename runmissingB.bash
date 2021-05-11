#for arg in 'softmax' 'no-softmax'; do srun -p cuda -x cuda01,cuda02 python train_mnist.py  --vary_name softmax --width 1000 --gd_mode 'full' "--$arg" & done
[ -z $max_run ] && max_run=10;
dir='slurm/scripts'
template='template.sbatch'
sbname="B-m"
#width=$2
models=$@
f=2
name=fraction-4
#incr=1
#sbname="a-c5-2"
fname="$dir/$sbname.sbatch"
cp "$dir/$template" $fname
exit_code=0
#((incr++))
#c=0.36
#ns=square
#date=`date +%y%m%d`  # format yymmdd
#root=results/$dataset/$date
file_missing="$1"

sed -i "s%^\(#SBATCH -J\) test_slurm%\1 $sbname%" $fname
sed -i "s%^\(#SBATCH -o .*/\).*%\1/$sbname.out%" $fname
sed -i "s%^\(#SBATCH -e .*/\).*%\1/$sbname.err%" $fname

#for shuffle in `seq 0 0.1 1`; 


while read line; do
    #model="$root_model$rid/checkpoint.pth"
    root_model=`echo $line | sed  "s-\(.*[0-9]\{6\}/\).*-\1-"`
    dataset=`echo $line | sed "s/.*\(mnist\|cifar10\).*/\1/"`
    id_run=`echo $line | sed "s/.*run-\([0-9]\+\).*/\1/"`
    W=`echo $line | sed "s/.*W-\([0-9]\+\).*/\1/"`
    L=`echo $line | sed "s/.*L-\([0-9]\+\).*/\1/"`
    #echo $W 
    [[ $W -lt 200 ]] && lr="0.001" || lr="0.01"
    #echo $lr
    #echo "root: $root_model, dataset: $dataset, id_run: $id_run"
    #echo "#srun piython train_fcn.py -oroot $root_model --name L-$L/W-$W/run-$id_run --width $W --depth $L --dataset $dataset -lr $lr -lrm manual"  >> $fname
    echo "#srun python eval_copy.py --model $root_model/L-$L/W-$W/run-$id_run/checkpoint.pth --name ds-f2 --fraction 2 --steps 200 --optim_mult"  >> $fname
done < $file_missing



    #for el in $els; do
        #echo "#srun python check_seq.py --model $model --learning_rate 0.003 --entry_layer $el --name f-2 --fraction 2 -lrm manual -lrs 0"  >> $fname
    #done
#for kr in `seq 0.1 0.1 0.5` 
#for ns in square

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

