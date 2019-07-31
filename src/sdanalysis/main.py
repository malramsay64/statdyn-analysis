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

import click
import yaml

from .molecules import Dimer, Disc, Sphere, Trimer
from .read import process_file
from .relaxation import compute_relaxations
from .version import __version__

logger = logging.getLogger(__name__)

gsd_logger = logging.getLogger("gsd")
gsd_logger.setLevel(logging.WARNING)
freud_logger = logging.getLogger("freud")
freud_logger.setLevel(logging.WARNING)

MOLECULE_OPTIONS = {"trimer": Trimer, "disc": Disc, "sphere": Sphere, "dimer": Dimer}


def _verbosity(_, __, value) -> None:
    levels = {0: "WARNING", 1: "INFO", 2: "DEBUG"}
    log_level = levels.get(value, "DEBUG")
    logging.basicConfig(level=log_level)
    logger.debug("Setting log level to %s", log_level)


def _file_logger(_, __, value) -> None:
    filename = value
    logging.basicConfig(filename=filename, level=logger.level)


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
    "--log-file",
    callback=_file_logger,
    expose_value=False,
    is_eager=True,
    help="Set output file for logging information.",
)
@click.option(
    "-s",
    "--num-steps",
    "--steps-max",
    "steps_max",
    type=int,
    help="Maximum number of steps for analysis",
)
@click.option(
    "--keyframe-interval",
    "--gen-steps",
    "keyframe_interval",  # save to the variable keyframe_interval
    type=int,
    default=1_000_000,
    help="Steps between key frames in simulation",
)
@click.option(
    "--linear-steps",
    type=int,
    default=100,
    help="Number of steps between exponential increase.",
)
@click.option(
    "--keyframe-max",
    "--max-gen",
    "keyframe_max",
    type=int,
    default=500,
    help="Maximum number of key frames",
)
@click.option(
    "--molecule",
    type=click.Choice(MOLECULE_OPTIONS),
    help="Molecule to use for simulation",
)
@click.option(
    "--wave-number",
    type=float,
    default=0.3,
    help="This is the wave number corresponding to the maximum of the structure factor",
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
    ctx.obj = {key: val for key, val in kwargs.items() if val is not None}


@sdanalysis.command()
@click.pass_obj
@click.option(
    "--mol-relaxations",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a file defining all the molecular relaxations to compute.",
)
@click.option(
    "--linear-dynamics",
    type=bool,
    default=False,
    is_flag=True,
    help="Flag to specify the configurations in a trajectory have linear steps between them.",
)
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("outfile", type=click.Path(file_okay=True, dir_okay=False))
def comp_dynamics(obj, mol_relaxations, linear_dynamics, infile, outfile) -> None:
    """Compute dynamic properties for a single input file."""
    outfile = Path(outfile)
    infile = Path(infile)

    # Create output directory where it doesn't already exists
    outfile.parent.mkdir(parents=True, exist_ok=True)

    if linear_dynamics:
        obj["linear_steps"] = None
    if mol_relaxations is not None:
        relaxations = yaml.parse(mol_relaxations)
    else:
        relaxations = None

    logger.debug("Processing: %s", infile)

    process_file(infile=infile, mol_relaxations=relaxations, outfile=outfile, **obj)


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


def _interactive_verbosity(_, __, value) -> None:
    if value:
        interactive_logger = logging.getLogger("sdanalysis.figures.interactive_config")
        interactive_logger.setLevel("DEBUG")
        logging.basicConfig(level="DEBUG")
        interactive_logger.debug("Setting log level of interactive_config to DEBUG")


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
@click.option(
    "-v",
    "--verbose",
    count=True,
    default=0,
    callback=_interactive_verbosity,
    expose_value=False,
    is_eager=True,
    help="Increase debug level",
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
