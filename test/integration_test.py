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

"""Testing that everything works together."""

from pathlib import Path

import pandas
import pytest

from sdanalysis.main import comp_dynamics, comp_relaxations


@pytest.fixture()
def trajectory():
    return Path(__file__).parent / "data/trajectory-Trimer-P13.50-T3.00.gsd"


def test_dynamics_file(dynamics_file):
    assert dynamics_file.is_file()
    with pandas.HDFStore(dynamics_file) as src:
        assert "/dynamics" in src.keys()
        assert "/molecular_relaxations" in src.keys()
        assert "/dynamics/" not in src.keys()


def test_runner_file(runner):
    import click

    @click.command()
    @click.argument("fname")
    def create_file(fname):
        with open(fname, "w") as dst:
            dst.write("testing\n")

    result = runner.invoke(create_file, ["test"])
    assert result.exit_code == 0
    assert (Path.cwd() / "test").is_file()


@pytest.mark.xfail
def test_dynamics(runner, trajectory):
    outdir = Path.cwd()
    print(outdir)
    result = runner.invoke(comp_dynamics, ["-o", str(outdir), str(trajectory)])
    print(list(outdir.glob("*")))
    assert result.exit_code == 0
    assert (outdir / "dynamics.h5").is_file()
    result = runner.invoke(comp_relaxations, ["dynamics.h5"])
    assert result.exit_code == 0, result.output


def test_relaxation_file(runner, dynamics_file):
    assert dynamics_file.is_file()
    result = runner.invoke(comp_relaxations, [str(dynamics_file)])
    assert result.exit_code == 0
    with pandas.HDFStore(dynamics_file) as src:
        assert "/relaxations" in src.keys()


@pytest.mark.xfail
def test_relaxation(runner, trajectory):
    outdir = Path.cwd()
    print(outdir)
    result = runner.invoke(comp_dynamics, ["-o", str(outdir), str(trajectory)])
    print(list(outdir.glob("*")))
    assert result.exit_code == 0
    assert (outdir / "dynamics.h5").is_file()
    result = runner.invoke(comp_relaxations, ["dynamics.h5"])
    assert result.exit_code == 0, result.output
