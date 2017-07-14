#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the simulation module."""

import os
from pathlib import Path

import hoomd
import pytest

from statdyn import crystals
from statdyn.simulation import initialise, simrun

from .crystal_test import CELL_DIMS

OUTDIR = Path('test/tmp')
OUTDIR.mkdir(exist_ok=True)


def test_run_npt():
    """Test an npt run."""
    snapshot = initialise.init_from_none()
    simrun.run_npt(snapshot,
                   context=hoomd.context.initialize(''),
                   temperature=3.00,
                   steps=10,
                   dynamics=False,
                   output=OUTDIR
                   )
    assert True


@pytest.mark.parametrize("max_initial", [1, 2])
def test_run_multiple_concurrent(max_initial):
    """Test running multiple concurrent."""
    snapshot = initialise.init_from_file(
        Path('test/data/Trimer-13.50-3.00.gsd')
    )
    simrun.run_npt(snapshot,
                   context=hoomd.context.initialize(''),
                   temperature=3.00,
                   steps=10,
                   max_initial=max_initial,
                   output=OUTDIR,
                   )
    assert True


def test_thermo():
    """Test the _set_thermo function works.

    There are many thermodynamic values set in the function and ensuring that
    they can all be initialised is crucial to a successful simulation.
    """
    output = Path('test/tmp')
    output.mkdir(exist_ok=True)
    snapshot = initialise.init_from_none()
    simrun.run_npt(snapshot,
                   context=hoomd.context.initialize(''),
                   temperature=3.00,
                   steps=10,
                   thermo_period=1,
                   output=OUTDIR
                   )
    assert True


@pytest.mark.parametrize("cell_dimensions", CELL_DIMS)
def test_orthorhombic_sims(cell_dimensions):
    """Test the initialisation from a crystal unit cell.

    This also ensures there is no unusual things going on with the calculation
    of the orthorhombic unit cell.
    """
    output = Path('test/tmp')
    output.mkdir(exist_ok=True)
    snap = initialise.init_from_crystal(crystals.TrimerP2(),
                                        cell_dimensions=cell_dimensions,
                                        )
    simrun.run_npt(snap,
                   context=hoomd.context.initialize(''),
                   temperature=0.1,
                   steps=10,
                   dynamics=False,
                   output=OUTDIR
                   )
    assert True


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
