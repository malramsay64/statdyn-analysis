#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from click.testing import CliRunner

from sdanalysis import SimulationParams, molecules, read, relaxation

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
    with TemporaryDirectory() as tmp:
        outfile = Path(tmp) / "dynamics.h5"
        sim_params = SimulationParams(infile=infile, outfile=outfile, output=tmp)
        read.process_file(sim_params)

        yield outfile
