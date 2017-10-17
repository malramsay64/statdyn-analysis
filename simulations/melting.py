#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Set running a series of simulations."""

import subprocess
import sys
from pathlib import Path

import numpy as np

pbs_file = """
#!/bin/bash
#PBS -N Trimer-P{P:.2f}-T{T:.2f}-I{I:.2f}
#PBS -m abe
#PBS -M malramsay64+quartz@gmail
#PBS -j oe
#PBS -o {outdir}/pbs-P{P:.2f}-T{T:.2f}-I{I:.2f}.log
#PBS -l select=1:ncpus={ncpus}
#PBS -l walltime=500:00:00
#PBS -l cput=3600:00:00
#PBS -S /bin/bash

export PATH=$HOME/.pyenv/versions/dev/bin:$PATH

mpirun -np {ncpus} sdrun create --pressure {P} -o {outdir} -t 0.2 --space-group p2 -s 1_000 --moment-inertia-scale {I} {outdir}/Trimer-P{P:.2f}-T0.2-I{I:.2f}-p2.gsd
mpirun -np {ncpus} sdrun equil --equil-type crys -t {T}  --pressure {P} --init-temp 0.2 -o {outdir} -s 100_000 --moment-inertia-scale {I} {outdir}/Trimer-P{P:.2f}-T0.2-I{I:.2f}-p2.gsd {outdir}/Trimer-P{P:.2f}-T{T:.2f}-I{I:.2f}-p2.gsd
mpirun -np {ncpus} sdrun prod -t {T} --pressure {P} -o {outdir} -s 10_000_000 --moment-inertia-scale {I} {outdir}/Trimer-P{P:.2f}-T{T:.2f}-I{I:.2f}-p2.gsd

"""


temperatures = np.arange(0.2, 1.6, 0.2)
pressures = np.arange(1.5, 13.5, 1.5)
mom_inertia = np.power(10., np.arange(-1, 3))

outdir = Path.home() / 'tmp1m/2017-10-12-phases'

if __name__ == "__main__":
    # ensure outdir exists
    outdir.mkdir(exist_ok=True)

    for T in temperatures:
        for P in pressures:
            for I in mom_inertia:
                cat_file = subprocess.Popen(
                    ['echo', pbs_file.format(T=T, P=P, I=I, outdir=outdir, ncpus=8)],
                    stdout=subprocess.PIPE)
                subprocess.Popen(['qsub'], stdin=cat_file.stdout, stdout=sys.stdout)
                cat_file.stdout.close()
