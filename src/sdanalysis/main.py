#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Run simulation with boilerplate taken care of by the statdyn library."""

import logging
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Tuple, cast

import click
import yaml

from .molecules import Dimer, Disc, Sphere, Trimer
from .params import SimulationParams
from .relaxation import compute_relaxations
from .threading import parallel_process_files
from .version import __version__

logger = logging.getLogger(__name__)

gsd_logger = logging.getLogger("gsd")
gsd_logger.setLevel(logging.WARNING)

MOLECULE_OPTIONS = {"trimer": Trimer, "disc": Disc, "sphere": Sphere, "dimer": Dimer}


def _verbosity(_, __, value) -> None:
    levels = {0: "WARNING", 1: "INFO", 2: "DEBUG"}
    log_level = levels.get(value, "DEBUG")
    logging.basicConfig(level=log_level)
    logger.debug(f"Setting log level to %s", log_level)


@click.group()
@click.version_option(__version__)
@click.option(
    "-v",
    "--verbose",
    count=True,
    default=0,
    callback=_verbosity,
    expose_value=False,
    is_eager=True,
    help="Increase debug level",
)
@click.option(
    "-s", "--num-steps", type=int, help="Maximum number of steps for analysis"
)
@click.option(
    "--keyframe-interval",
    "--gen-steps",
    "gen_steps",  # save to the variable gen_steps
    type=int,
    help="Steps between keyframes in simulation",
)
@click.option(
    "--linear-steps", type=int, help="Number of steps between exponential increase."
)
@click.option("--max-gen", type=int, help="Maximum number of keyframes")
@click.option(
    "--molecule",
    type=click.Choice(MOLECULE_OPTIONS),
    help="Molecule to use for simnulation",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Location to save all output files; required to be a directory.",
)
@click.pass_context
def sdanalysis(ctx, **kwargs) -> None:
    """Run main function."""
    logging.debug("Running main function")
    logging.debug("Creating SimulationParams with values:\n%s", pformat(kwargs))
    ctx.obj = SimulationParams(
        **{key: val for key, val in kwargs.items() if val is not None}
    )
    if ctx.obj.output is not None:
        ctx.obj.output.mkdir(exist_ok=True)


@sdanalysis.command()
@click.pass_obj
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Location to save all output files; required to be a directory.",
)
@click.option("--mol-relaxations", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--linear-dynamics",
    type=bool,
    default=False,
    is_flag=True,
    help="Flag to specify the configurations in a trajectory have linear steps between them.",
)
@click.argument(
    "infile", nargs=-1, type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def comp_dynamics(sim_params, output, mol_relaxations, linear_dynamics, infile) -> None:
    """Compute dynamic properties for a number of input files."""
    assert isinstance(infile, tuple)
    for i in infile:
        assert i is not None
        assert isinstance(i, str)
    if sim_params is None:
        sim_params = SimulationParams()
    if output is not None:
        sim_params.output = output
    sim_params.outfile = sim_params.output / "dynamics.h5"
    if linear_dynamics:
        sim_params.linear_steps = None
    if mol_relaxations is not None:
        relaxations = yaml.parse(mol_relaxations)
    else:
        relaxations = None

    # Create output directory where it doesn't already exists
    sim_params.output.mkdir(parents=True, exist_ok=True)

    logger.info("Processing: %s", infile)

    infile = cast(Tuple[str], infile)
    parallel_process_files(infile, sim_params, relaxations)


@sdanalysis.command()
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def comp_relaxations(infile) -> None:
    """Compute the summary time value for the dynamic quantities.

    This computes the characteristic timescale of the dynamic quantities which have been
    calculated and are present in INFILE. The INFILE is a path to the pre-computed
    dynamic quantities and needs to be in the HDF5 format with either the '.hdf5' or
    '.h5' extension.

    The output is written to the table 'relaxations' in INFILE.

    """
    logger.info("Processing: %s", infile)
    compute_relaxations(infile)


@sdanalysis.command()
@click.option("--ip", multiple=True, help="Allow connections from these locations.")
@click.option("--directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-m",
    "--model",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    multiple=True,
    help="Pickled scikit-learn Models to use for visualisation of machine learning algorithms.",
)
def figure(ip, directory, model) -> None:
    """Start bokeh server with the file passed."""
    from bokeh.server.server import Server
    from bokeh.application import Application
    from bokeh.application.handlers.function import FunctionHandler

    from .figures.interactive_config import make_document

    if isinstance(ip, str):
        ip = (ip,)
    if directory:
        directory = Path(directory)
        make_document = partial(make_document, directory=directory, models=model)

    apps = {"/": Application(FunctionHandler(make_document))}
    server = Server(apps, allow_websocket_origin=list(ip))
    server.run_until_shutdown()
    logger.info("Bokeh server terminated.")
