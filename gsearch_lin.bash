#for arg in 'softmax' 'no-softmax'; do srun -p cuda -x cuda01,cuda02 python train_mnist.py  --vary_name softmax --width 1000 --gd_mode 'full' "--$arg" & done
dir='slurm/scripts'
template='template.sbatch'
#dataset=$1
name=$1
shift
models=$@
max_run=3
incr=1
for model in $models;
do
    sbname=$name-$incr
    #sbname=$dataset
    fname="$dir/$sbname.sbatch"
    [ -z $max_run ] && max_run=3;
    cp "$dir/$template" $fname
    exit_code=0
    ((incr++))
    c=1
    #nt=$2
    date=`date +%y%m%d`  # format yymmdd
    root=results/$dataset/$date

    sed -i "s/^\(#SBATCH -J\) test_slurm/\1 $sbname/" $fname

    #for shuffle in `seq 0 0.1 1`;
    for kr in `seq 0.1 0.1 0.5` 
    do
        name=kr-$kr
        #echo "#srun python train_mnist.py -o $root --name $name --shuffle $shuffle --dataset $dataset --coefficient $c" >> $fname; 
        #echo "#srun python train_lin.py --model $model --keep_ratio $kr --name $name" >> $fname; 
    done;

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
done;

