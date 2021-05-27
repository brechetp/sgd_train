[ -z $max_run ] && max_run=1;
tp_dir='slurm/scripts'
template='template_rtx.sbatch'
sbname="bcw1"
fname="$tp_dir/$sbname.sbatch"
cp "$tp_dir/$template" $fname
#root=$1  # results/cifar10/210409/L-2
missing_file="missing_nets"
name="ds-f2_optim-mult"
models=$@
f=2
#for var in `seq 1 5`; do
for model in $models; do
    dir=`dirname $model`/$name
    fn_log_model="`dirname $model`/logs.txt"
    if [ ! -f "$fn_log_model" ]; then
        echo "$fn_log_model does not exist, skipping"
        continue;
    fi
    #lr=`sed -n 's/^learning_rate= \([0-9.]*\)$/\1/p' $fn_log_model`  # the previous learning rate
    #model=$root/var-$var/checkpoint.pth
    #for el in `seq 0 2`; do
        #(( $el == 0 )) && lr=0.01 || lr=0.01
        #lr=0.0005
        file=$dir/eval_copy.pth
        #echo $file
        #echo $var $el `seq 1 20`>> $missing_file
        if [ ! -f "$file" ]; then
            #echo "#srun python check_seq.py --model $model --fraction $f --entry_layer $el --name $name  -lr $lr -lrs 0  --min_epochs 100 --nepochs 400 ">> $fname; 
            echo "#srun python eval_copy.py --model $model --optim_mult --name ds-f2 --steps 200"  >> $fname
        else
            echo $file
        fi;
#        else  # some draws exist
#            fn_log=$dir/logs_entry_$el.txt
#            if [ -f $fn_log ]; then
#                #for dn in `seq 1 20`; do  # record the draw index to the file
#                let l=`wc -l $fn_log | cut -d' ' -f1`
#                #(( $l < 100 )) && echo $fn_log
#                let n=`(( $l < 100 )) && echo $l || tail -n 2 $fn_log | head -n 1 | cut -d' ' -f2 | cut -d',' -f1` 
#                lr_prev=`sed -n 's/^learning_rate= \([0-9.]*\)$/\1/p' $fn_log_model`  # the previous learning rate
#                if (( $(echo "$lr_prev != $lr" | bc -l) )); then  # not the correct learning rate
#                    ##echo $lr_prev $lr
#                    echo "#srun python check_seq.py --model $model --fraction $f --entry_layer $el --name $name  -lr $lr -lrs 0  --min_epochs 100 --nepochs 400 ">> $fname; 
#                else
#                #echo $n
#                    if (( $n < 100 )); then   # could be that there is only logs.txt
#                        chpt=$file
#                        echo "#srun python check_seq.py --checkpoint $file --fraction $f --entry_layer $el --name $name --min_epochs 100  -lr $lr -lrs 0  --nepochs 400 ">> $fname; 
#                    fi;
#                fi;
#            fi;
#        fi;
    done;

        #done;

cat $fname;



if (( $max_run > 0 )); then
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
    sed -i "`expr $i - $max_run + 1`,${i}s/^#*//" $fname
    #
fi
