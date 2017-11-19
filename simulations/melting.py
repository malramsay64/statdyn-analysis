#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Set running a series of simulations."""

import itertools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

pbs_file = """#!/usr/bin/env python
#PBS -N Trimer
#PBS -m abe
#PBS -M malramsay64+quartz@gmail
#PBS -j oe
#PBS -o {outdir}/pbs.log
#PBS -l select=1:ncpus={ncpus}
#PBS -l walltime=500:00:00
#PBS -l cput=3600:00:00
#PBS -V
{array_flag}

import subprocess
import os
from pathlib import Path

def create_fname(mol, temp, pressure, mom_I, crys):
    return mol + '-P' + pressure + '-T' + temp + '-I' + mom_I + '-' + crys + '.gsd'

all_values = {values}

job_index = int(os.environ.get('PBS_ARRAY_INDEX', 0))

temperature, pressure, moment_inertia, crystal = all_values[job_index]

temperature = '{{:.2f}}'.format(temperature)
pressure = '{{:.2f}}'.format(pressure)
moment_inertia = '{{:.2f}}'.format(moment_inertia)

ncpus = {ncpus}

common_opts = [
    '--pressure', pressure,
    '--temperature', temperature,
    '--moment-inertia-scale', moment_inertia,
    '--output', '{outdir}',
]

run_comand = ['sdrun']
if ncpus > 1:
    run_comand = [
        'mpirun',
        '--np', str(ncpus),
    ] + run_comand

create_temp = '0.2'
create_out = '{outdir}' + create_fname('Trimer', create_temp, pressure, moment_inertia, crystal)

create_opts = [
    '--space-group', crystal,
    '--steps', '{create_steps}',
    '--temperature', create_temp,
    create_out,
]

subprocess.call(run_comand + ['create'] + common_opts + create_opts)

equil_out = '{outdir}' + create_fname('Trimer', temperature, pressure, moment_inertia, crystal)

equil_opts = [
    '--equil-type', 'crystal',
    '--init-temp', create_temp,
    '--steps', '{equil_steps}',
    create_out,
    equil_out,
]

subprocess.call(run_comand + ['equil'] + common_opts + equil_opts)

prod_opts = [
    '--steps', '{prod_steps}',
    equil_out,
]

subprocess.call(run_comand + ['prod'] + common_opts + prod_opts)

"""


temperatures = np.arange(0.2, 0.4, 0.2)
pressures = np.arange(1.5, 4.5, 1.5)
mom_inertia = np.power(10., np.arange(0, 1))
crystals = ['p2']

outdir = Path.home() / 'tmp1m/2017-10-17-test2'

if __name__ == "__main__":
    # ensure outdir exists
    outdir.mkdir(exist_ok=True)

    all_values = list(itertools.product(temperatures, pressures, mom_inertia, crystals))

    def get_array_flag(num_values: int) -> str:
        if num_values == 1:
            return ''
        else:
            return f'#PBS -J 0-{num_values-1}'

    sub_file = pbs_file.format(
        values=all_values,
        array_flag=get_array_flag(len(all_values)),
        outdir=outdir,
        create_steps=1000,
        equil_steps=100_000,
        prod_steps=10_000_000,
        ncpus=8
    )

    subprocess.run(['qsub'],
                   input=sub_file,
                   stdout=sys.stdout,
                   stderr=sys.stderr,
                   env=os.environ,
                   )

    with open(outdir / 'sub_file.py', 'w') as tf:
        tf.write(sub_file)
