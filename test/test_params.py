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
from hypothesis import example, given
from hypothesis.strategies import text

from statdyn.crystals import TrimerP2
from statdyn.molecules import Dimer, Disc, Molecule, Sphere, Trimer
from statdyn.simulation.params import SimulationParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

SIM_PARAMS = SimulationParams()

MOLECULE_LIST = [Molecule, Sphere, Trimer, Dimer, Disc, None]

class setValues(object):
    """Temporarily set values for testing.

    This is a context manager that can be used to temporarily set the values of a
    SimulationParams instance. This simplifies the setup allowing for a single global
    instance that is modified with every test. The modifications also make it clear
    what is actually being tested.

    """

    def __init__(self, sim_params: SimulationParams, **kwargs) -> None:
        """Initialise setValues class.

        Args:
            sim_params (class:`statdyn.simulation.params.SimulationParams`): The
                instance that is to be temporarily modified.

        Kwargs:
            key: value

        Any of the keys and values that are held by a SimulationParams instance.
        """
        self.params = sim_params
        self.modifications = kwargs
        self.original = {key: sim_params.parameters.get(key)
                         for key in kwargs.keys()
                         if sim_params.parameters.get(key) is not None}

    def __enter__(self) -> SimulationParams:
        for key, val in self.modifications.items():
            self.params.parameters[key] = val
        logger.debug('Parameter on entry %s', str(self.params.parameters))
        return self.params

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for key, _ in self.modifications.items():
            try:
                del self.params.parameters[key]
            except AttributeError:
                pass
        self.params.parameters.update(self.original)
        logger.debug('Parameter on exit %s', str(self.params.parameters))


@given(key=text(), value=text())
@example(key='outfile', value='test')
@example(key='outfile_path', value='testing')
def test_setvalues(key, value):
    """Ensure setValues sets value correctly and returns to previous state.

    This is just testing that the values in the dictionary are the same since
    that is what the setValues class is modifying. Note that this isn't testing
    the getattr, setattr, delattr of the SimulationParams class."""
    test_values = deepcopy(SIM_PARAMS.parameters)
    with setValues(SIM_PARAMS, **{key: value}) as sim_params:
        assert sim_params.parameters.get(key) == value
    assert test_values == sim_params.parameters


@pytest.mark.parametrize('mol', MOLECULE_LIST)
def test_molecule(mol):
    with setValues(SIM_PARAMS, molecle=mol):
        assert SIM_PARAMS.molecle == mol


def test_default_molecule():
    assert SIM_PARAMS.molecule == Trimer()


def test_mol_crys():
    crys = TrimerP2()
    with setValues(SIM_PARAMS, crystal=crys):
        assert SIM_PARAMS.molecule == crys.molecule


@pytest.mark.parametrize('outfile', [
    'test/data',
    Path('test/data')
])
def test_outfile(outfile):
    with setValues(SIM_PARAMS, outfile=outfile):
        assert SIM_PARAMS.parameters.get('outfile') == outfile
        assert str(outfile) == SIM_PARAMS.outfile


@pytest.mark.parametrize('outfile_path', [
    'test/output',
    Path('test/output')
])
def test_outdir(outfile_path):
    with setValues(SIM_PARAMS, outfile_path=outfile_path):
        assert SIM_PARAMS.parameters.get('outfile_path') == outfile_path
        assert Path(outfile_path) == SIM_PARAMS.outfile_path
