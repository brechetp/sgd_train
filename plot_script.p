plot data_file using 1:2 with lines title ""
set xlabel "Depth (layers)"
set ylabel "Maximum Removed Units (#neurons)"
set output out_file 
replot

