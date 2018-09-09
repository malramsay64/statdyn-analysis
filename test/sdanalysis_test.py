#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Test the sdrun command line tools."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from click.testing import CliRunner
from tables import open_file

from sdanalysis.main import comp_relaxations


@pytest.fixture
def runner():
    runner = CliRunner()
    with runner.isolated_filesystem():
        yield runner


@pytest.fixture
def output_directory():
    with TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_comp_dynamics(output_directory):
    command = [
        "sdanalysis",
        "-v",
        "-o",
        output_directory,
        "comp_dynamics",
        "test/data/trajectory-Trimer-P13.50-T3.00.gsd",
    ]
    ret = subprocess.run(command)
    assert ret.returncode == 0


class TestCompRelaxations:
    def test_missing_file(self, runner):
        result = runner.invoke(comp_relaxations, ["fail"])
        assert result.exit_code != 0
        print(result.output)
        assert isinstance(result.exception, SystemExit)

    def test_incorrect_extension(self, runner):
        Path("fail").write_text("text")
        result = runner.invoke(comp_relaxations, ["fail"])
        assert result.exit_code != 0
        print(result.output)
        assert isinstance(result.exception, ValueError)

    def test_non_hdf5_file(self, runner):
        Path("fail.hdf5").write_text("text")
        result = runner.invoke(comp_relaxations, ["fail.hdf5"])
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)

    def test_missing_dynamics(self, runner):
        infile = Path("fail.hdf5")
        with open_file(str(infile), "w") as dst:
            pass
        result = runner.invoke(comp_relaxations, [str(infile)])
        assert result.exit_code != 0
        assert isinstance(result.exception, KeyError)
        assert "'dynamics'" in str(result.exception)

    def test_missing_mol_relaxations(self, runner):
        infile = Path("fail.hdf5")
        with open_file(str(infile), "w") as dst:
            dst.create_group("/", "dynamics")
        result = runner.invoke(comp_relaxations, [str(infile)])
        assert result.exit_code != 0
        assert isinstance(result.exception, KeyError)
        assert "'dynamics'" not in str(result.exception)
        assert "'molecular_relaxations'" in str(result.exception)
