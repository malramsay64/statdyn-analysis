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


@pytest.fixture(params=[2.90, None])
def obj(request):
    """Default values for the statdyn analysis command line."""
    return {
        "keyframe_interval": 1_000_000,
        "keyframe_max": 500,
        "wave_number": request.param,
    }


@pytest.fixture(
    params=[
        "data/trajectory-Trimer-P13.50-T3.00.gsd",
        "data/short-time-variance.lammpstrj",
    ]
)
def infile(request):
    """A path to an input file which can processed.

    This includes all the types of files which are supported.

    """
    return Path(__file__).parent / request.param


@pytest.fixture()
def infile_gsd():
    """The path to a gsd file which can be used for input.

    This is to test all the analysis which uses the gsd files as input.
    """
    return Path(__file__).parent / "data/trajectory-Trimer-P13.50-T3.00.gsd"


@pytest.fixture()
def infile_lammps():
    """The path to a lammps file which can be used for input.

    This is to test all the analysis which uses the lammps files as input.
    """
    return Path(__file__).parent / "data/short-time-variance.lammpstrj"


@pytest.fixture()
def frame(infile):
    return next(read.open_trajectory(infile))


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
