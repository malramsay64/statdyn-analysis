#!/bin/bash

file=TrimerEquil.pbs
scratch_dir=$HOME/project/hoomd2/equil_00
dir=$(pwd)

(cd "$scratch_dir" || exit ; qsub "$dir"/"$file")
