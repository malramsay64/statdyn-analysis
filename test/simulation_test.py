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
from pathlib import Path

import hoomd
import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers, tuples

from statdyn import crystals
from statdyn.simulation import equilibrate, initialise, simrun
from statdyn.simulation.helper import SimulationParams

OUTDIR = Path('test/tmp')
OUTDIR.mkdir(exist_ok=True)

PARAMETERS = SimulationParams(
    temperature=0.4,
    num_steps=100,
    crystal=crystals.TrimerP2(),
    outfile_path=Path('test/tmp'),
)


@pytest.mark.simulation
def test_run_npt():
    """Test an npt run."""
    snapshot = initialise.init_from_none()
    simrun.run_npt(
        snapshot=snapshot,
        context=hoomd.context.initialize(''),
        sim_params=PARAMETERS,
        dynamics=False,
    )
    assert True


@given(integers(max_value=10, min_value=1))
@settings(max_examples=5, timeout=0)
@pytest.mark.hypothesis
def test_run_multiple_concurrent(max_initial):
    """Test running multiple concurrent."""
    snapshot = initialise.init_from_file(
        Path('test/data/Trimer-13.50-3.00.gsd')
    )
    simrun.run_npt(
        snapshot,
        context=hoomd.context.initialize(''),
        sim_params=PARAMETERS,
        max_initial=max_initial,
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
    snapshot = initialise.init_from_none()
    simrun.run_npt(
        snapshot,
        context=hoomd.context.initialize(''),
        sim_params=PARAMETERS,
    )
    assert True


@given(tuples(integers(max_value=30, min_value=5),
              integers(max_value=5, min_value=1)))
@settings(max_examples=10, timeout=0)
def test_orthorhombic_sims(cell_dimensions):
    """Test the initialisation from a crystal unit cell.

    This also ensures there is no unusual things going on with the calculation
    of the orthorhombic unit cell.
    """
    cell_dimensions = cell_dimensions[0], cell_dimensions[1]*6
    output = Path('test/tmp')
    output.mkdir(exist_ok=True)
    snap = initialise.init_from_crystal(
        PARAMETERS,
        cell_dimensions=cell_dimensions,
    )
    snap = equilibrate.equil_crystal(
        snap,
        sim_params=PARAMETERS,
    )
    simrun.run_npt(
        snap,
        context=hoomd.context.initialize(''),
        sim_params=PARAMETERS,
        dynamics=False,
    )
    assert True
    gc.collect()


@pytest.mark.xfail
def test_file_placement():
    """Ensure files are located in the correct directory when created."""
    outdir = Path('test/output')
    outdir.mkdir(exist_ok=True)
    current = list(Path('.').glob('*'))
    for i in outdir.glob('*'):
        os.remove(str(i))
    snapshot = initialise.init_from_none()
    simrun.run_npt(snapshot,
                   hoomd.context.initialize(''),
                   sim_params=PARAMETERS,
                   dynamics=True,
                   )
    assert current == list(Path('.').glob('*'))
    assert (outdir / 'Trimer-13.50-3.00.gsd').is_file()
    assert (outdir / 'dump-13.50-3.00.gsd').is_file()
    assert (outdir / 'thermo-13.50-3.00.log').is_file()
    assert (outdir / 'trajectory-13.50-3.00.gsd').is_file()
