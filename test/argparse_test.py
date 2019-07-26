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

from sdanalysis import main
from sdanalysis.main import MOLECULE_OPTIONS, comp_dynamics, sdanalysis
from sdanalysis.version import __version__

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def dummy_process_file(infile, **kwargs):
    print(f"infile={infile}")
    for key, value in kwargs.items():
        print(f"{key}={value}")


@click.command("dummy_subcommand")
@click.pass_obj
def dummy_subcommand(obj):
    for key, value in obj.items():
        print(f"{key}={value}")


sdanalysis.add_command(dummy_subcommand, "dummy_subcommand")


def test_version(runner):
    result = runner.invoke(sdanalysis, ["--version"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output


@pytest.mark.parametrize("arg", ["-v", "-vv", "-vvv", "-vvvv", "--verbose"])
def test_verbose(runner, arg):
    result = runner.invoke(sdanalysis, [arg, "dummy_subcommand"])
    assert result.exit_code == 0, result.output


def test_comp_dynamics_infile(monkeypatch, runner, outfile, obj):
    monkeypatch.setattr(main, "process_file", dummy_process_file)

    # Check error on non-existent file
    result = runner.invoke(comp_dynamics, ["non-existent.file", str(outfile)], obj=obj)
    assert result.exit_code == 2, result.output + result.stderr
    assert 'Invalid value for "infile": Path "non-existent.file" does not exist.'

    datafile = Path(__file__).parent / "data/trajectory-Trimer-P13.50-T3.00.gsd"
    result = runner.invoke(comp_dynamics, [str(datafile), str(outfile)], obj=obj)
    assert result.exit_code == 0, result.output
    print(result.output)
    assert "infile=None" not in result.output
    assert f"infile={datafile}" in result.output


def test_comp_dynamics_mol_relax(runner, obj):
    # Check error on non-existent file
    result = runner.invoke(
        comp_dynamics, ["--mol-relaxations", "non-existent.file"], obj=obj
    )
    assert result.exit_code == 2, result.output
    assert (
        'Invalid value for "mol_relaxations": Path "non-existent.file" does not exist.'
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
            for value in MOLECULE_OPTIONS:
                yield {"option": option, "value": value}
        else:
            for value in [0, 100, 1000, 10000]:
                yield {"option": option, "value": value}


@pytest.mark.parametrize("params", create_params())
def test_sdanalysis_options(runner, params):
    result = runner.invoke(
        sdanalysis, [params["option"], params["value"], "dummy_subcommand"]
    )
    assert result.exit_code == 0, result.output
    logger.debug("Command Output: \n%s", result.output)
    option = params["option"].strip("-").replace("-", "_")
    assert f"{option}={params['value']}" in result.output


@pytest.mark.parametrize("output", ["output", "subdir/output"])
def test_sdanalysis_output(runner, output):
    result = runner.invoke(sdanalysis, ["--output", output, "dummy_subcommand"])
    assert result.exit_code == 0, result.output
    assert Path(output).exists
    assert f"output={output}" in result.output


def test_create_output_dir(runner, outfile, obj):
    infile = Path(__file__).parent / "data/trajectory-Trimer-P13.50-T3.00.gsd"
    result = runner.invoke(comp_dynamics, [str(infile), str(outfile)], obj=obj)
    assert result.exit_code == 0, result.output


def test_figure():
    pass
