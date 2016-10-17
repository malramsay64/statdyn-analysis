#!/bin/bash

file=TrimerEquil.pbs
scratch_dir=$HOME/project/hoomd2/equil_00
dir=$(pwd)

temps=(5.00 4.00 3.50 3.00 2.50 2.00 1.80 1.60 1.50 1.40 1.35 1.30 1.25 1.20 1.15 1.10)

mkdir -p "$scratch_dir"

jobid=
for t in "${temps[@]}"; do
    if [[ -e $jobid ]]; then
        jobid=$(cd "$scratch_dir" || exit; qsub -v temp="$t" -W afterok:"$jobid" "$dir"/"$file")
    else
        jobid=$(cd "$scratch_dir" || exit; qsub -v temp="$t" "$dir"/"$file")
    fi

done
