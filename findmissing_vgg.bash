
[ -z $max_run ] && max_run=1;
tp_dir='slurm/scripts'
template='template_v100.sbatch'
sbname="ctcv"
fname="$tp_dir/$sbname.sbatch"
cp "$tp_dir/$template" $fname
root=$1  # results/cifar10/210409/L-2
missing_file="missing_nets"
name="f2"
f=2
for var in 5; do
    dir=$root/var-$var/$name
    model=$root/var-$var/checkpoint.pth
    for el in `seq 0 10`; do
        path=$dir/entry_$el
        #echo $var $el `seq 1 20`>> $missing_file
        if [ ! -d $path ]; then
            echo "#srun python annex_vgg.py --model $model --fraction $f --entry_layer $el --name $name --draws 20 -lrm manual -lr 0.002 -lrs 0  --nepochs 1000 ">> $fname; 
        else  # some draws exist
            #for dn in `seq 1 20`; do  # record the draw index to the file
            let n=`ls $dir/entry_$el | grep checkpoint_ | cut -d'_' -f3 | cut -d'.' -f1 | sort -n | tail -n 1`
            if (( $n == 0 )); then   # could be that there is only logs.txt
                echo "#srun python annex_vgg.py --model $model --fraction $f --entry_layer $el --name $name --draws 20 -lrm manual -lr 0.002 -lrs 0  --nepochs 1000 ">> $fname; 
            else
                if (( $n < 20 && $el > 0 )); then
                    chkpt="$path/checkpoint_draw_$n.pth"
                    echo "#srun python annex_vgg.py --checkpoint $chkpt --fraction $f" >> $fname; 
                fi;
            fi;
        fi;
    done;
done;

cat $fname;



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
    #sleep 1
    sed -i "$i,`expr $i + $max_run - 1`s/^/#/" $fname
    let i=$i+$max_run
done;

