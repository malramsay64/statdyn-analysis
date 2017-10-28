#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import itertools
import os
import subprocess
from pathlib import Path
from typing import List


pbs_header = """#!/usr/bin/env python
#PBS -N {name}
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

def create_fname(mol, temp, pressure='13.50', mom_I='1.00', crys=None, ext='.gsd'):
    fname = mol + '-P' + pressure + '-T' + temp + '-I' + mom_I
    if crys:
        fname += '-' + crys
    fname += ext
    return fname

all_values = {values}

job_index = int(os.environ.get('PBS_ARRAY_INDEX', 0))

temperature, pressure, moment_inertia, crystal = all_values[job_index]

temperature = '{{:.2f}}'.format(temperature)
pressure = '{{:.2f}}'.format(pressure)
moment_inertia = '{{:.2f}}'.format(moment_inertia)

ncpus = {ncpus}

outdir = Path('{outdir}')

common_opts = [
    '--pressure', pressure,
    '--temperature', temperature,
    '--moment-inertia-scale', moment_inertia,
    '--space-group', crystal,
    '--output', str(outdir),
    '--steps', '{steps}',
]

run_comand = ['sdrun']
if ncpus > 1:
    run_comand = [
        'mpirun',
        '--np', str(ncpus),
    ] + run_comand

"""

create_liquid = pbs_header + """

create_out = outdir / create_fname('Trimer', temperature, pressure, moment_inertia)

create_opts = [
    '--space-group', 'pg',
    '--lattice-lengths', '25', '25',
    create_out,
]

if not create_out.exists():
    return_code = subprocess.call(run_comand + ['create'] + common_opts + create_opts)
    assert return_code == 0

"""

create_crys = pbs_header + """

create_out = outdir / create_fname('Trimer', temperature, pressure, moment_inertia)

create_opts = [
    create_out,
]

if not create_out.exists():
    return_code = subprocess.call(run_comand + ['create'] + common_opts + create_opts)
    assert return_code == 0

"""

equilibrate = pbs_header + """

create_temp = '{create_temp:.2f}'

create_out = outdir / create_fname('Trimer', create_temp, pressure, moment_inertia)
equil_out = outdir / create_fname('Trimer', temperature, pressure, moment_inertia)

equil_opts = [
    '--equil-type', '{equil_type}',
    '--init-temp', create_temp,
    create_out,
    equil_out,
]

if not equil_out.exists():
    return_code = subprocess.call(run_comand + ['equil'] + common_opts + equil_opts)
    assert return_code == 0

"""

production = pbs_header + """

equil_out = outdir / create_fname('Trimer', temperature, pressure, moment_inertia)

prod_opts = [
    '{dynamics}',
    equil_out,
]

return_code = subprocess.call(run_comand + ['prod'] + common_opts + prod_opts)
assert return_code == 0

"""



def get_array_flag(num_values: int) -> str:
    if num_values == 1:
        return ''
    else:
        return f'#PBS -J 0-{num_values-1}'

temperatures = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0, 1.5, 1.8]
pressures = [1.]
mom_inertia = [1.]
crystals: List[str] = [None]

outdir = Path.home() / 'tmp1m/2017-10-27-dynamics'

if __name__ == "__main__":
    # ensure outdir exists
    outdir.mkdir(exist_ok=True)

    create_temp = 1.80
    create_values = list(itertools.product([create_temp], pressures, mom_inertia, crystals))
    create_pbs = create_liquid.format(
        name='dynamics-create',
        values=create_values,
        array_flag=get_array_flag(len(create_values)),
        outdir=outdir,
        steps=100_000,
        ncpus=8,
    )
    create_process = subprocess.run(
        ['qsub'],
        input=create_pbs,
        stdout=subprocess.PIPE,
        env=os.environ,
    )
    with open(outdir / 'create_pbs.py', 'w') as tf:
        tf.write(create_pbs)

    assert create_process.returncode == 0

    all_values = list(itertools.product(temperatures, pressures, mom_inertia, crystals))
    equil_pbs = equilibrate.format(
        name='dynamics-equil',
        equil_type='liquid',
        create_temp=create_temp,
        values=all_values,
        array_flag=get_array_flag(len(all_values)),
        outdir=outdir,
        steps=10_000_000,
        ncpus=8,
    )
    equil_process = subprocess.run(
        ['qsub', '-W', 'depend=afterok:'+create_process.stdout],
        input=equil_pbs,
        stdout=subprocess.PIPE,
        env=os.environ,
    )
    with open(outdir / 'equil_pbs.py', 'w') as tf:
        tf.write(equil_pbs)

    assert equil_process.returncode == 0

    prod_pbs = production.format(
        name='dynamics-prod',
        dynamics='--dynamics',
        values=all_values,
        array_flag=get_array_flag(len(all_values)),
        outdir=outdir,
        steps=100_000_000,
        ncpus=8,
    )
    prod_process = subprocess.run(
        ['qsub', '-W', 'depend=afterok:'+create_process.stdout],
        input=prod_pbs,
        stdout=subprocess.PIPE,
        env=os.environ,
    )
    with open(outdir / 'prod_pbs.py', 'w') as tf:
        tf.write(prod_pbs)

    assert prod_process.returncode == 0
