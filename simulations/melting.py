#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Set running a series of simulations."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

pbs_file = """#!/usr/bin/env python
#PBS -N Trimer-P{P:.2f}-T{T:.2f}-I{I:.2f}
#PBS -m abe
#PBS -M malramsay64+quartz@gmail
#PBS -j oe
#PBS -o {outdir}/pbs-P{P:.2f}-T{T:.2f}-I{I:.2f}.log
#PBS -l select=1:ncpus={ncpus}
#PBS -l walltime=500:00:00
#PBS -l cput=3600:00:00
#PBS -V

import subprocess
import os
from pathlib import Path

temperature = {T}
pressure = {P}
moment_inertia = {I}
ncpus = {ncpus}

common_opts = [
    '--pressure', str(pressure),
    '--temperature', str(temperature),
    '--moment-inertia-scale', str(moment_inertia),
    '--output', '{outdir}',
]

run_comand = ['sdrun']
if ncpus > 1:
    run_comand = [
        'mpirun',
        '--np', str(ncpus),
    ] + run_comand

create_out = '{outdir}/Trimer-P{P:.2f}-T0.2-I{I:.2f}-p2.gsd'

create_opts = [
    '--space-group', 'p2',
    '--steps', '1000',
    '--temperature', '0.2',
    create_out,
]

subprocess.call(run_comand + ['create'] + common_opts + create_opts)

equil_out = '{outdir}/Trimer-P{P:.2f}-T{T:.2f}-I{I:.2f}-p2.gsd'

equil_opts = [
    '--equil-type', 'crystal',
    '--init-temp', '0.2',
    '--steps', '100000',
    create_out,
    equil_out,
]

subprocess.call(run_comand + ['equil'] + common_opts + equil_opts)

prod_opts = [
    '--steps', '10000000',
    equil_out,
]

subprocess.call(run_comand + ['prod'] + common_opts + prod_opts)

"""


temperatures = np.arange(0.2, 0.4, 0.2)
pressures = np.arange(1.5, 3.0, 1.5)
mom_inertia = np.power(10., np.arange(0, 1))

outdir = Path.home() / 'tmp1m/2017-10-17-testing'

if __name__ == "__main__":
    # ensure outdir exists
    outdir.mkdir(exist_ok=True)



    for T in temperatures:
        for P in pressures:
            for I in mom_inertia:
                cat_file = subprocess.Popen(
                    ['echo', pbs_file.format(T=T, P=P, I=I, outdir=outdir, ncpus=8, version='dev', home=Path.home())],
                    stdout=subprocess.PIPE)

                subprocess.Popen(['qsub'],
                                 stdin=cat_file.stdout,
                                 stdout=sys.stdout,
                                 stderr=sys.stderr,
                                 env=os.environ,
                                 )

                with open('testfile.py', 'w') as tf:
                    tf.write(pbs_file.format(T=T, P=P, I=I, outdir=outdir, ncpus=8, version='dev', home=Path.home()))
                cat_file.stdout.close()
