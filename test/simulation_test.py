#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the simulation module."""

import gc
import os
import subprocess
from pathlib import Path

import gsd.hoomd
import hoomd
import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers, tuples

from statdyn import crystals
from statdyn.simulation import equilibrate, initialise, simrun
from statdyn.simulation.params import SimulationParams, paramsContext

OUTDIR = Path('test/tmp')
OUTDIR.mkdir(exist_ok=True)
HOOMD_ARGS="--mode=cpu"

PARAMETERS = SimulationParams(
    temperature=0.4,
    num_steps=100,
    crystal=crystals.TrimerP2(),
    outfile_path=Path('test/tmp'),
    outfile='test/tmp/testout',
    dynamics=False,
    hoomd_args=HOOMD_ARGS
)


@pytest.mark.simulation
def test_run_npt():
    """Test an npt run."""
    snapshot = initialise.init_from_none(hoomd_args=HOOMD_ARGS)
    simrun.run_npt(
        snapshot=snapshot,
        context=hoomd.context.initialize(''),
        sim_params=PARAMETERS,
    )
    assert True


@given(integers(max_value=10, min_value=1))
@settings(max_examples=5, deadline=1000)
@pytest.mark.hypothesis
def test_run_multiple_concurrent(max_initial):
    """Test running multiple concurrent."""
    snapshot = initialise.init_from_file(
        Path('test/data/Trimer-13.50-3.00.gsd'),
        hoomd_args=HOOMD_ARGS,
    )
    with paramsContext(PARAMETERS, max_initial=max_initial):
        simrun.run_npt(snapshot,
                       context=hoomd.context.initialize(''),
                       sim_params=PARAMETERS
                       )
    assert True
    gc.collect()


def test_thermo():
    """Test the _set_thermo function works.

    There are many thermodynamic values set in the function and ensuring that
    they can all be initialised is crucial to a successful simulation.
    """
    output = Path('test/tmp')
    output.mkdir(exist_ok=True)
    snapshot = initialise.init_from_none(hoomd_args=HOOMD_ARGS)
    simrun.run_npt(
        snapshot,
        context=hoomd.context.initialize(''),
        sim_params=PARAMETERS,
    )
    assert True


@given(tuples(integers(max_value=30, min_value=5),
              integers(max_value=5, min_value=1)))
@settings(max_examples=10, deadline=2000)
def test_orthorhombic_sims(cell_dimensions):
    """Test the initialisation from a crystal unit cell.

    This also ensures there is no unusual things going on with the calculation
    of the orthorhombic unit cell.
    """
    cell_dimensions = cell_dimensions[0], cell_dimensions[1]*6
    output = Path('test/tmp')
    output.mkdir(exist_ok=True)
    with paramsContext(PARAMETERS, cell_dimensions=cell_dimensions):
        snap = initialise.init_from_crystal(PARAMETERS)
    snap = equilibrate.equil_crystal(snap, sim_params=PARAMETERS)
    simrun.run_npt(snap,
                   context=hoomd.context.initialize(''),
                   sim_params=PARAMETERS,
                   )
    assert True
    gc.collect()


def test_equil_file_placement():
    outdir = Path('test/output')
    outfile = outdir / 'test_equil'
    current = list(Path.cwd().glob('*'))
    for i in outdir.glob('*'):
        os.remove(str(i))
    with paramsContext(PARAMETERS, outfile_path=outdir, outfile=outfile, temperature=4.00):
        snapshot = initialise.init_from_none(hoomd_args=HOOMD_ARGS)
        equilibrate.equil_liquid(snapshot, PARAMETERS)
        assert current == list(Path.cwd().glob('*'))
        assert Path(outfile).is_file()
    for i in outdir.glob('*'):
        os.remove(str(i))


def test_file_placement():
    """Ensure files are located in the correct directory when created."""
    outdir = Path('test/output')
    current = list(Path.cwd().glob('*'))
    for i in outdir.glob('*'):
        os.remove(str(i))
    with paramsContext(PARAMETERS, outfile_path=outdir, dynamics=True, temperature=3.00):
        snapshot = initialise.init_from_none(hoomd_args=HOOMD_ARGS)
        simrun.run_npt(snapshot, hoomd.context.initialize(''), sim_params=PARAMETERS)
        assert current == list(Path.cwd().glob('*'))
        assert (outdir / 'Trimer-P13.50-T3.00.gsd').is_file()
        assert (outdir / 'dump-Trimer-P13.50-T3.00.gsd').is_file()
        assert (outdir / 'thermo-Trimer-P13.50-T3.00.log').is_file()
        assert (outdir / 'trajectory-Trimer-P13.50-T3.00.gsd').is_file()
    for i in outdir.glob('*'):
        os.remove(str(i))


@pytest.mark.parametrize('pressure, temperature', [(1.0, 1.8), (13.5, 3.00)])
def test_interface(pressure, temperature):
    init_temp = 0.4
    create_command = [
        'sdrun', 'create',
        '--pressure', '{}'.format(pressure),
        '--space-group', 'p2',
        '--lattice-lengths', '48', '42',
        '--temperature', '{}'.format(init_temp),
        '--steps', '1000',
        '--output', OUTDIR,
        '-vvv',
        '--hoomd-args', '"--mode=cpu"',
        str(OUTDIR / 'create_interface-P{:.2f}-T{:.2f}.gsd'.format(pressure, init_temp)),
    ]
    melt_command = [
        'sdrun', 'equil',
        '--equil-type', 'interface',
        '--pressure', '{}'.format(pressure),
        '--space-group', 'p2',
        '--temperature', '{}'.format(temperature),
        '--output', OUTDIR,
        '--steps', '1000',
        '-vvv',
        '--hoomd-args', '"--mode=cpu"',
        str(OUTDIR / 'create_interface-P{:.2f}-T{:.2f}.gsd'.format(pressure, init_temp)),
        str(OUTDIR / 'melt_interface-P{:.2f}-T{:.2f}.gsd'.format(pressure, temperature)),
    ]
    create = subprocess.run(create_command)
    assert create.returncode == 0
    melt = subprocess.run(melt_command)
    assert melt.returncode == 0

def test_dynamics_output():
    """Ensure files are located in the correct directory when created."""
    outdir = Path('test/output')
    for i in outdir.glob('*'):
        os.remove(str(i))
    with paramsContext(PARAMETERS, outfile_path=outdir, dynamics=True, temperature=3.00):
        snapshot = initialise.init_from_none(hoomd_args=HOOMD_ARGS)
        simrun.run_npt(snapshot, hoomd.context.initialize(''), sim_params=PARAMETERS)
        assert (outdir / 'trajectory-Trimer-P13.50-T3.00.gsd').is_file()
        with gsd.hoomd.open(str(outdir / 'trajectory-Trimer-P13.50-T3.00.gsd')) as trj:
            assert [f.configuration.step for f in trj] == list(range(101))
    for i in outdir.glob('*'):
        os.remove(str(i))
