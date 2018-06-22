#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the SimulationParams class."""

import logging
from pathlib import Path

import pytest

from sdanalysis.molecules import Dimer, Disc, Molecule, Sphere, Trimer
from sdanalysis.params import SimulationParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def sim_params():
    return SimulationParams(num_steps=1000, temperature=1.0, space_group="p2")


@pytest.fixture(params=[Molecule, Sphere, Trimer, Dimer, Disc])
def molecule(request):
    return request.param()


def test_default_molecule(sim_params):
    assert sim_params.molecule == Trimer()


def test_molecule(sim_params, molecule):
    with sim_params.temp_context(molecle=molecule):
        assert sim_params.molecle == molecule


@pytest.mark.parametrize("outfile", ["test/data", Path("test/data")])
def test_outfile(sim_params, outfile):
    with sim_params.temp_context(outfile=outfile):
        logger.debug("sim_params.outfile = %s", sim_params.outfile)
        assert sim_params.outfile == Path(outfile)


@pytest.mark.parametrize("output", ["test/output", Path("test/output")])
def test_outdir(sim_params, output):
    with sim_params.temp_context(output=output):
        logger.debug("sim_params.output = %s", sim_params.output)
        assert sim_params.output == Path(output)


def func(sim_params, value):
    return getattr(sim_params, value)


def test_function_passing(sim_params):
    assert sim_params.num_steps == 1000
    with sim_params.temp_context(num_steps=2000):
        assert func(sim_params, "num_steps") == 2000
        assert sim_params.num_steps == 2000
    assert func(sim_params, "num_steps") == 1000
    assert sim_params.num_steps == 1000
