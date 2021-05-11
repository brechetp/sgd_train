
root=$1  # results/cifar10/210409/L-2
find $root/W-* -mindepth 1 -maxdepth 1 -type d \( -exec sh -c 'cd "$0"; find . \( -name . -o -prune \) -name "ds-f2_optim-mult" | grep -q .' {} \; -o -print \)

