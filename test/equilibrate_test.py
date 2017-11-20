#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the equilibrate module."""

from pathlib import Path

from statdyn.crystals import TrimerPg
from statdyn.simulation import equilibrate, initialise, params

SIM_PARAMS = params.SimulationParams(
    temperature=0.4,
    num_steps=100,
    crystal=TrimerPg(),
    outfile_path=Path('test/tmp'),
    cell_dimensions=(10, 10),
    outfile=Path('test/tmp/out.gsd'),
)


def init_frame():
    return initialise.init_from_crystal(SIM_PARAMS)


def test_equil_crystal():
    equilibrate.equil_crystal(init_frame(), SIM_PARAMS)
    assert True


def test_equil_interface():
    equilibrate.equil_interface(init_frame(), SIM_PARAMS)
    assert True


def test_equil_liquid():
    equilibrate.equil_liquid(init_frame(), SIM_PARAMS)
    assert True
