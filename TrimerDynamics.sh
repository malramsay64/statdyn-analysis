#!/bin/bash

file=TrimerDynamics.pbs
scratch_dir=$HOME/project/hoomd2/prod_00
dir=$(pwd)

mkdir -p "$scratch_dir"

temps=(5.00 4.00 3.50 3.00 2.50 2.00 1.80 1.60 1.50 1.40 1.35 1.30 1.25 1.20 1.15 1.10)

(cd "$scratch_dir" || exit ; qsub "$dir"/"$file")
