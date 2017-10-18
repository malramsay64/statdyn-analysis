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
from pathlib import Path

create_file = """#!/usr/bin/env python
#PBS -N create_interface
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
    '--space-group', crystal,
    '--output', '{outdir}',
]

run_comand = ['sdrun']
if ncpus > 1:
    run_comand = [
        'mpirun',
        '--np', str(ncpus/2),
    ] + run_comand

init_temp = '0.39'

create_out = '{outdir}/' + create_fname('Trimer', init_temp, pressure, moment_inertia, crystal)

create_opts = [
    '--space-group', crystal,
    '--steps', '1000',
    '--temperature', init_temp,
    '--lattice-lengths', '48', '42',
    create_out,
]

if not Path(create_out).exists():
    print(' '.join(run_comand + ['create'] + common_opts + create_opts))
    return_code = subprocess.call(run_comand + ['create'] + common_opts + create_opts)
    assert return_code == 0

melt_out = '{outdir}/' + create_fname('Trimer', '{create_temp}', pressure, moment_inertia, crystal)

melt_opts = [
    '--space-group', crystal,
    '--steps', '{create_steps}',
    '--equil-type', 'interface',
    '--temperature', '{create_temp}',
    create_out,
    melt_out,
]

if not Path(melt_out).exists():
    print(' '.join(run_comand + ['equil'] + common_opts + melt_opts))
    return_code = subprocess.call(run_comand + ['equil'] + common_opts + melt_opts)
    assert return_code == 0

"""

pbs_file = """#!/usr/bin/env python
#PBS -N interface_production
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
    '--space-group', crystal,
    '--output', '{outdir}',
]

run_comand = ['sdrun']
if ncpus > 1:
    run_comand = [
        'mpirun',
        '--np', str(ncpus),
    ] + run_comand


create_out = '{outdir}/' + create_fname('Trimer', '{create_temp}', pressure, moment_inertia, crystal)

equil_out = '{outdir}/' + create_fname('Trimer', temperature, pressure, moment_inertia, crystal)

equil_opts = [
    '--equil-type', 'interface',
    '--init-temp', '{create_temp}',
    '--steps', '{equil_steps}',
    create_out,
    equil_out,
]

subprocess.call(run_comand + ['equil'] + common_opts + equil_opts)

prod_opts = [
    '--steps', '{prod_steps}',
    '--no-dynamics',
    equil_out,
]

subprocess.call(run_comand + ['prod'] + common_opts + prod_opts)

"""


temperatures = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0, 1.5]
pressures = [1.]
mom_inertia = [1., 10., 100., 1000.]
crystals = ['p2', 'pg', 'p2gg']

outdir = Path.home() / 'tmp1m/2017-10-18-interface'

if __name__ == "__main__":
    # ensure outdir exists
    outdir.mkdir(exist_ok=True)

    all_values = list(itertools.product(temperatures, pressures, mom_inertia, crystals))

    create_temp = 1.80
    create_values = list(itertools.product([create_temp], pressures, mom_inertia, crystals))

    def get_array_flag(num_values: int) -> str:
        if num_values == 1:
            return ''
        else:
            return f'#PBS -J 0-{num_values-1}'

    create_pbs = create_file.format(
        values=create_values,
        create_temp=create_temp,
        array_flag=get_array_flag(len(create_values)),
        outdir=outdir,
        create_steps=100_000,
        ncpus=8,
    )

    prod_pbs = pbs_file.format(
        values=all_values,
        array_flag=get_array_flag(len(all_values)),
        outdir=outdir,
        create_temp=create_temp,
        equil_steps=1_000_000,
        prod_steps=100_000_000,
        ncpus=8,
    )

    create_process = subprocess.run(
        ['qsub'],
        input=create_pbs,
        encoding='utf-8',
        stdout=subprocess.PIPE,
    )

    job_id = create_process.stdout
    print(job_id)

    subprocess.run(['qsub', '-W', 'depend=afterok:'+job_id],
                   input=prod_pbs,
                   encoding='utf-8',
                   env=os.environ,
                   )

    with open(outdir / 'create_pbs.py', 'w') as tf:
        tf.write(create_pbs)

    with open(outdir / 'prod_pbs.py', 'w') as tf:
        tf.write(prod_pbs)
