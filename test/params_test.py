#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the SimulationParams class."""

import logging
from copy import deepcopy
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis.strategies import text

from sdanalysis.molecules import Dimer, Disc, Molecule, Sphere, Trimer
from sdanalysis.params import SimulationParams, paramsContext

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

SIM_PARAMS = SimulationParams(num_steps=1000, temperature=1.0, space_group='p2')

MOLECULE_LIST = [Molecule, Sphere, Trimer, Dimer, Disc, None]


@given(key=text(), value=text())
@example(key='outfile', value='test')
@example(key='outfile_path', value='testing')
@settings(deadline=None)
def test_paramContext(key, value):
    """Ensure paramsContext sets value correctly and returns to previous state.

    This is just testing that the values in the dictionary are the same since
    that is what the paramsContext class is modifying. Note that this isn't testing
    the getattr, setattr, delattr of the SimulationParams class.

    """
    test_values = deepcopy(SIM_PARAMS.parameters)
    with paramsContext(SIM_PARAMS, **{key: value}) as sim_params:
        assert sim_params.parameters.get(key) == value
    assert test_values == sim_params.parameters


@pytest.mark.parametrize('mol', MOLECULE_LIST)
def test_molecule(mol):
    with paramsContext(SIM_PARAMS, molecle=mol):
        assert SIM_PARAMS.molecle == mol


def test_default_molecule():
    assert SIM_PARAMS.molecule == Trimer()


@pytest.mark.parametrize('outfile', [
    'test/data',
    Path('test/data')
])
def test_outfile(outfile):
    with paramsContext(SIM_PARAMS, outfile=outfile):
        assert SIM_PARAMS.parameters.get('outfile') == outfile
        assert str(outfile) == SIM_PARAMS.outfile


@pytest.mark.parametrize('outfile_path', [
    'test/output',
    Path('test/output')
])
def test_outdir(outfile_path):
    with paramsContext(SIM_PARAMS, outfile_path=outfile_path):
        assert SIM_PARAMS.parameters.get('outfile_path') == outfile_path
        assert Path(outfile_path) == SIM_PARAMS.outfile_path


def func(sim_params, value):
    return getattr(sim_params, value)


@pytest.mark.parametrize('sim_params', [SIM_PARAMS])
def test_function_passing(sim_params):
    assert sim_params.num_steps == 1000
    with paramsContext(sim_params, num_steps=2000):
        assert func(sim_params, 'num_steps') == 2000
        assert sim_params.num_steps == 2000
    assert func(sim_params, 'num_steps') == 1000
    assert sim_params.num_steps == 1000

