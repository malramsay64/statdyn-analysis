#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the parsing of arguments gives the correct results."""

import logging
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import sdanalysis as sdanalysis_module
from sdanalysis.main import MOLECULE_OPTIONS, comp_dynamics, sdanalysis
from sdanalysis.params import SimulationParams
from sdanalysis.version import __version__

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def print_params_values(sim_params: SimulationParams) -> None:
    for key, value in sim_params.__dict__.items():
        print(f"{key}={value}")


def dummy_process_file(sim_params: SimulationParams, compute_relaxations):
    print_params_values(sim_params)


# Monkey patch process file function
sdanalysis_module.main.process_file = dummy_process_file


@sdanalysis.command()
@click.pass_obj
def dummy_subcommand(obj):
    print_params_values(obj)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sim_params():
    return SimulationParams()


def test_version(runner):
    result = runner.invoke(sdanalysis, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.mark.parametrize("arg", ["-v", "-vv", "-vvv", "-vvvv", "--verbose"])
def test_verbose(runner, arg):
    result = runner.invoke(sdanalysis, [arg, "dummy_subcommand"])
    assert result.exit_code == 0


def test_comp_dynamics_infile(runner, sim_params):
    # Check error on nonexistant file
    with runner.isolated_filesystem():
        result = runner.invoke(comp_dynamics, ["nonexistant.file"], obj=sim_params)
        assert result.exit_code == 2
        assert 'Invalid value for "infile": Path "nonexistant.file" does not exist.'

    datafile = "test/data/trajectory-Trimer-P13.50-T3.00.gsd"
    result = runner.invoke(comp_dynamics, [datafile], obj=sim_params)
    assert result.exit_code == 0
    print(result.output)
    assert "_infile=None" not in result.output
    assert f"_infile={datafile}" in result.output


def test_comp_dynamics_mol_relax(runner, sim_params):
    # Check error on nonexistant file
    with runner.isolated_filesystem():
        result = runner.invoke(
            comp_dynamics, ["--mol-relaxations", "nonexistant.file"], obj=sim_params
        )
        assert result.exit_code == 2
        assert (
            'Invalid value for "mol_relaxations": Path "nonexistant.file" does not exist.'
        )


def create_params():
    for option in [
        "--num-steps",
        "--gen-steps",
        "--linear-steps",
        "--max-gen",
        "--molecule",
    ]:
        value = None

        if "molecule" in option:
            for value in MOLECULE_OPTIONS.keys():
                yield {"option": option, "value": value}
        else:
            for value in [0, 100, 1000, 10000]:
                yield {"option": option, "value": value}


@pytest.mark.parametrize("params", create_params())
def test_sdanalysis_options(runner, params):
    result = runner.invoke(
        sdanalysis, [params["option"], params["value"], "dummy_subcommand"]
    )
    assert result.exit_code == 0
    logger.debug("Command Output: \n%s", result.output)
    option = params["option"].strip("-").replace("-", "_")
    assert f"{option}={params['value']}" in result.output


@pytest.mark.parametrize("output", ["output", "output"])
def test_sdanalysis_output(runner, output):
    with runner.isolated_filesystem():
        result = runner.invoke(sdanalysis, ["--output", output, "dummy_subcommand"])
        assert result.exit_code == 0
        assert Path(output).exists
        assert f"_output={output}" in result.output


def test_figure():
    pass
