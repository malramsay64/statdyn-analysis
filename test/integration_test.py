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


def test_dynamics_file(dynamics_file):
    """Ensure the structure of the dynamics file is consistent."""
    assert dynamics_file.is_file()
    with pandas.HDFStore(dynamics_file) as src:
        assert "/dynamics" in src.keys()
        assert "/molecular_relaxations" in src.keys()
        assert "/dynamics/" not in src.keys()


def test_runner_file(runner):
    """Test to ensure the runner fixture is performing as expected."""
    import click

    @click.command()
    @click.argument("fname")
    def create_file(fname):
        with open(fname, "w") as dst:
            dst.write("testing\n")

    result = runner.invoke(create_file, ["test"])
    assert result.exit_code == 0
    assert (Path.cwd() / "test").is_file()


def test_comp_dynamics(runner, infile_gsd, outfile, obj):
    """Ensure the comp_dynamics command is working."""
    result = runner.invoke(comp_dynamics, [str(infile_gsd), str(outfile)], obj=obj)
    assert result.exit_code == 0
    assert outfile.is_file()
    with pandas.HDFStore(outfile) as src:
        assert "/dynamics" in src.keys()
        assert "/molecular_relaxations" in src.keys()


def test_comp_relaxations(runner, dynamics_file):
    """Ensure the comp_relaxations command is working."""
    assert dynamics_file.is_file()
    result = runner.invoke(comp_relaxations, [str(dynamics_file)])
    assert result.exit_code == 0
    with pandas.HDFStore(dynamics_file) as src:
        assert "/relaxations" in src.keys()


def test_entire_process(runner, infile_gsd, outfile, obj):
    """Ensure the combination of comp_dynamics and comp_relaxations works.

    These two functions are intended to be used after each other, so this ensures that
    they continue to function in concert.

    """
    # Dynamics Calculation
    result = runner.invoke(comp_dynamics, [str(infile_gsd), str(outfile)], obj=obj)
    assert result.exit_code == 0
    assert outfile.is_file()
    with pandas.HDFStore(outfile) as src:
        assert "/dynamics" in src.keys()
        assert "/molecular_relaxations" in src.keys()

    # Relaxation Calculation
    result = runner.invoke(comp_relaxations, [str(outfile)])
    assert result.exit_code == 0, result.output
    with pandas.HDFStore(outfile) as src:
        assert "/relaxations" in src.keys()
        assert "/dynamics" in src.keys()
        assert "/molecular_relaxations" in src.keys()
