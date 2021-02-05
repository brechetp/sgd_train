#!/bin/bash
# takes root as first argumen
root=${1%/}  # remove trailing /
output_file=$root/parse_results.data
echo "#Depth    Removed units" > $output_file
for fname in $(find $root -iname 'bounds.txt' | sort -t'L' -k2 -n -r); 
do
    rs=$(set -o pipefail; grep "stop" -B 1 $fname | tail -n 2 | head -n 1 | cut -d " " -f2 ||
        tail -n 1 $fname | cut -d " " -f5)
    #echo $rs
    NL=${fname#$root/L-}  # removes from the beginning
    NL=${NL%/*}  # removes the tail
    echo -e "$NL\t$rs" >> $output_file
done


#gnuplot -e "data_file='$output_file'; out_file='${output_file%.data/.pdf}'" plot_script.p
