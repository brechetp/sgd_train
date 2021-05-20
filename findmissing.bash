
roots=$@  # results/cifar10/210409/L-2/W-..
#for el in `seq 0 10`; do
    #for draw in `seq 1 20`; do
    for root in $roots; do
    find $root -mindepth 1 -maxdepth 1 -type d \( -exec sh -c 'cd "$0"; find . \( -name . -o -prune \) -name "" | grep -q .' {} \; -o -print \)
done

