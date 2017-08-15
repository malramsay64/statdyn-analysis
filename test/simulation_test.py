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

OUTDIR = Path('test/tmp')
OUTDIR.mkdir(exist_ok=True)


@pytest.mark.simulation
def test_run_npt():
    """Test an npt run."""
    snapshot = initialise.init_from_none()
    simrun.run_npt(
        snapshot,
        context=hoomd.context.initialize(''),
        temperature=3.00,
        steps=100,
        dynamics=False,
        output=OUTDIR
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
        temperature=3.00,
        steps=1000,
        max_initial=max_initial,
        output=OUTDIR,
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
        temperature=3.00,
        steps=100,
        thermo_period=1,
        output=OUTDIR
    )
    assert True


@given(tuples(integers(max_value=30, min_value=5),
              integers(max_value=30, min_value=5)))
@settings(max_examples=10, timeout=0)
def test_orthorhombic_sims(cell_dimensions):
    """Test the initialisation from a crystal unit cell.

    This also ensures there is no unusual things going on with the calculation
    of the orthorhombic unit cell.
    """
    output = Path('test/tmp')
    output.mkdir(exist_ok=True)
    snap = initialise.init_from_crystal(
        crystals.TrimerP2(),
        cell_dimensions=cell_dimensions,
    )
    snap = equilibrate.equil_crystal(
        snap,
        equil_temp=0.5,
        equil_steps=100,
    )
    simrun.run_npt(
        snap,
        context=hoomd.context.initialize(''),
        temperature=0.5,
        steps=100,
        dynamics=False,
        output=OUTDIR
    )
    assert True
    gc.collect()


def test_file_placement():
    """Ensure files are located in the correct directory when created."""
    outdir = Path('test/output')
    outdir.mkdir(exist_ok=True)
    current = list(Path('.').glob('*'))
    _ = [os.remove(i) for i in outdir.glob('*')]
    snapshot = initialise.init_from_none()
    simrun.run_npt(snapshot,
                   hoomd.context.initialize(''),
                   temperature=3.00,
                   steps=10,
                   dynamics=True,
                   max_initial=1,
                   output=outdir
                   )
    assert current == list(Path('.').glob('*'))
    assert (outdir / 'Trimer-13.50-3.00.gsd').is_file()
    assert (outdir / 'dump-13.50-3.00.gsd').is_file()
    assert (outdir / 'thermo-13.50-3.00.log').is_file()
    assert (outdir / 'trajectory-13.50-3.00.gsd').is_file()
