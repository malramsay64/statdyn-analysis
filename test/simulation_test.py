#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Test the simulation module
"""

from statdyn import Simulation, initialise
import pytest

def test_run_npt():
    snapshot = initialise.init_from_none().take_snapshot()
    Simulation.run_npt(snapshot, 3.00, 100)
    assert True


def test_run_npt_from_file():
    snapshot = initialise.init_from_file(
        'test/data/Trimer-13.50-3.00.gsd').take_snapshot()
    Simulation.run_npt(snapshot, 3.00, 100)
    assert True


def test_run_multiple_concurrent():
    snapshot = initialise.init_from_file(
        'test/data/Trimer-13.50-3.00.gsd').take_snapshot()
    Simulation.run_multiple_concurrent(snapshot, 3.00, 100)
