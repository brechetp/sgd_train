#!/bin/bash

# the first argument is the data file to plot with gnuplot
data_in="$1"
echo $data_in
# the second argument is the output file otherwise use the first argument
plot_out=$2
: ${plot_out:=${data_in%.data/.pdf}}

gnuplot <<- EOF
set output "${plot_out}"
plot "${data_in}" with linespoints "Removed neurons"
set xlabel "Depth (layers)"
set ylabel "Maximum Removed Units (#neurons)"
EOF
