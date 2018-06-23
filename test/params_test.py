#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the SimulationParams class."""

import logging
from itertools import product
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


def create_params():
    all_params = [
        "molecule",
        "pressure",
        "temperature",
        "moment_inertia_scale",
        "harmonic_force",
        "space_group",
    ]
    space_groups = ["p2", "p2gg", "pg"]
    molecules = [Trimer()]
    values1 = [None, 0, 0.1, 1]
    values2 = [0, 0.1, 1]
    for space_group, molecule, value1, value2 in product(
        space_groups, molecules, values1, values2
    ):
        params = {}
        for param in all_params:
            if param == "molecule":
                params["molecule"] = molecule
            elif param == "space_group":
                params["space_group"] = space_group
            elif param in ["temperature", "pressure"]:
                params[param] = value2
            else:
                params[param] = value1
        yield params


def get_filename_prefix(key):
    prefixes = {
        "temperature": "T",
        "pressure": "P",
        "moment_inertia_scale": "I",
        "harmonic_force": "K",
    }
    return prefixes.get(key, "")


@pytest.mark.parametrize("params", create_params())
def test_filename(sim_params, params):

    with sim_params.temp_context(**params):
        fname = sim_params.filename().stem
    for key, value in params.items():
        if value is not None:
            prefix = get_filename_prefix(key)
            assert isinstance(prefix, str)
            logger.debug("Prefix: %s, Value: %s", prefix, value)
            if isinstance(value, (int, float)):
                assert f"{prefix}{value:.2f}" in fname
            else:
                assert f"{prefix}{value}" in fname
