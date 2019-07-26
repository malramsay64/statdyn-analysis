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

from sdanalysis import molecules, read

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


@pytest.fixture
def obj():
    return {"keyframe_interval": 1_000_000, "keyframe_max": 500, "wave_number": 2.90}


@pytest.fixture()
def infile_gsd():
    return Path(__file__).parent / "data/trajectory-Trimer-P13.50-T3.00.gsd"


@pytest.fixture()
def outfile():
    """The Path object of a temporary output file."""
    with TemporaryDirectory() as output:
        yield Path(output) / "test"


@pytest.fixture()
def dynamics_file(infile_gsd, obj):
    """A temporary file for which the dynamics quantities have been calculated."""
    with TemporaryDirectory() as tmp:
        outfile = Path(tmp) / "test.h5"
        read.process_file(infile_gsd, outfile=outfile, **obj)
        yield outfile
