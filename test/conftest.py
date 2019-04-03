#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
#
# pylint: disable=redefined-outer-name
#

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from click.testing import CliRunner

from sdanalysis import SimulationParams, molecules
from sdanalysis.threading import parallel_process_files

MOLECULE_LIST = [
    molecules.Molecule,
    molecules.Trimer,
    molecules.Dimer,
    molecules.Disc,
    molecules.Sphere,
]


@pytest.fixture(scope="module", params=MOLECULE_LIST)
def mol(request):
    return request.param()


@pytest.fixture
def runner():
    r = CliRunner()
    with r.isolated_filesystem():
        yield r


@pytest.fixture()
def dynamics_file():
    infile = Path(__file__).parent / "data/trajectory-Trimer-P13.50-T3.00.gsd"
    with TemporaryDirectory() as output:
        outfile = Path(output) / "dynamics.h5"
        sim_params = SimulationParams(outfile=outfile, output=output)
        parallel_process_files([infile], sim_params)

        yield outfile


@pytest.fixture()
def sim_params():
    with TemporaryDirectory() as output:
        params = SimulationParams(output=output)
        yield params
