#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
#
# pylint: disable=no-self-use, unused-argument
#

"""Test the sdrun command line tools."""

from pathlib import Path

from tables import open_file

from sdanalysis.main import comp_dynamics, comp_relaxations, sdanalysis


class TestSdanalysis:
    def test_verbosity_info(self, runner):
        result = runner.invoke(sdanalysis, ["-v"])
        assert result.exit_code == 2, result.output
        assert "DEBUG" not in result.output


class TestCompDynamics:

    datafile = Path(__file__).parent / "data/trajectory-Trimer-P13.50-T3.00.gsd"

    def test_datafile(self, runner):
        print(self.datafile)
        assert self.datafile.is_file()

    def test_missing_file(self, runner):
        result = runner.invoke(comp_dynamics, ["fail", "output"])
        assert result.exit_code != 0, result.output
        assert isinstance(result.exception, SystemExit)

    def test_real_data(self, runner, obj):
        result = runner.invoke(comp_dynamics, [str(self.datafile), "output"], obj=obj)
        assert result.exit_code == 0, result.output


class TestCompRelaxations:
    def test_missing_file(self, runner):
        result = runner.invoke(comp_relaxations, ["fail"])
        assert result.exit_code != 0, result.output
        print(result.output)
        assert isinstance(result.exception, SystemExit)

    def test_incorrect_extension(self, runner):
        Path("fail").write_text("text")
        result = runner.invoke(comp_relaxations, ["fail"])
        assert result.exit_code != 0, result.output
        assert isinstance(result.exception, ValueError)

    def test_non_hdf5_file(self, runner):
        Path("fail.hdf5").write_text("text")
        result = runner.invoke(comp_relaxations, ["fail.hdf5"])
        assert result.exit_code != 0, result.output
        assert isinstance(result.exception, ValueError)

    def test_missing_dynamics(self, runner):
        infile = Path("fail.hdf5")
        with open_file(str(infile), "w"):
            pass
        result = runner.invoke(comp_relaxations, [str(infile)])
        assert result.exit_code != 0, result.output
        assert isinstance(result.exception, KeyError)
        assert "'dynamics'" in str(result.exception)

    def test_missing_mol_relaxations(self, runner):
        infile = Path("fail.hdf5")
        with open_file(str(infile), "w") as dst:
            dst.create_group("/", "dynamics")
        result = runner.invoke(comp_relaxations, [str(infile)])
        assert result.exit_code != 0, result.output
        assert isinstance(result.exception, KeyError)
        assert "'dynamics'" not in str(result.exception)
        assert "'molecular_relaxations'" in str(result.exception)
