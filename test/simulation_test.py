#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the simulation module."""

import pytest
from statdyn import Simulation, initialise


def test_run_npt():
    """Test an npt run."""
    snapshot = initialise.init_from_none().take_snapshot()
    Simulation.run_npt(snapshot, 3.00, 100, dyn_many=False)
    assert True


@pytest.mark.parametrize("dyn_many", [True, False])
def test_run_multiple_concurrent(dyn_many):
    """Test running multiple concurrent."""
    snapshot = initialise.init_from_file(
        'test/data/Trimer-13.50-3.00.gsd').take_snapshot()
    Simulation.run_npt(snapshot, 3.00, 100, dyn_many=dyn_many)
    assert True


def test_thermo():
    """Test the _set_thermo function works.

    There are many thermodynamic values set in the function and ensuring that
    they can all be initialised is crucial to a successful simulation.
    """
    snapshot = initialise.init_from_none().take_snapshot()
    Simulation.run_npt(snapshot, 3.00, 100, thermo=True, thermo_period=1)
    assert True
