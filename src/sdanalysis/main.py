#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Run simulation with boilerplate taken care of by the statdyn library."""

import logging
from pathlib import Path
from pprint import pformat

import click
import numpy as np
import pandas
import yaml
from scipy.stats import hmean

from .molecules import Dimer, Disc, Sphere, Trimer
from .params import SimulationParams
from .read import process_file
from .relaxation import compute_relaxation_value, translate_relaxation
from .version import __version__

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

gsd_logger = logging.getLogger("gsd")
gsd_logger.setLevel(logging.WARNING)

MOLECULE_OPTIONS = {"trimer": Trimer, "disc": Disc, "sphere": Sphere, "dimer": Dimer}


def _verbosity(_, __, value) -> None:
    root_logger = logging.getLogger("statdyn")
    levels = {0: "WARNING", 1: "INFO", 2: "DEBUG"}
    log_level = levels.get(value, "DEBUG")
    print(f"value: {value}, log_level: {log_level}")
    logging.basicConfig(level=log_level)
    root_logger.setLevel(log_level)
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
@click.option("--gen-steps", type=int, help="Steps between keyframes in simulation")
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
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def comp_dynamics(sim_params, output, mol_relaxations, linear_dynamics, infile) -> None:
    """Compute dynamic properties."""
    if sim_params is None:
        sim_params = SimulationParams()
    if output is not None:
        sim_params.output = output
    sim_params.infile = infile
    if linear_dynamics:
        sim_params.linear_steps = None
    if mol_relaxations is not None:
        compute_relaxations = yaml.parse(mol_relaxations)
    else:
        compute_relaxations = None
    process_file(sim_params, compute_relaxations)


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
    from tables import open_file, HDF5ExtError

    infile = Path(infile)
    # Check is actually an HDF5 file
    try:
        with open_file(str(infile)):
            pass
    except HDF5ExtError:
        raise ValueError("The argument 'infile' requires an hdf5 input file.")

    # Check input file contains the tables required
    with open_file(str(infile)) as src:
        if "/dynamics" not in src:
            raise KeyError(
                "Table 'dynamics' not found in input file,"
                " try rerunning `sdanalysis comp_dynamics`."
            )
        if "/molecular_relaxations" not in src:
            raise KeyError(
                f"Table 'molecular_relaxations' not found in input file,"
                " try rerunning `sdanalysis comp_dynamics`."
            )

    assert infile.suffix in [".hdf5", ".h5"]
    df_dyn = pandas.read_hdf(infile, "dynamics")
    # Remove columns with no relaxation value to calculate
    df_dyn.drop(
        ["mean_displacement", "mean_rotation", "mfd", "overlap", "start_index"],
        inplace=True,
    )
    # Average over all initial times
    df_dyn = df_dyn.groupby(["time", "temperature", "pressure"]).mean()

    relaxations = df_dyn.groupby(["temperature", "pressure"]).aggregate(
        compute_relaxation_value
    )
    relaxations.columns = [
        translate_relaxation(quantity) for quantity in relaxations.columns
    ]

    df_mol = pandas.read_hdf(infile, "molecular_relaxations")
    df_mol.replace(2 ** 32 - 1, np.nan, inplace=True)
    df_mol.index.names = ["init_frame", "molecule"]
    df_mol = df_mol.groupby(["init_frame", "temperature", "pressure"]).agg(np.mean)
    df_mol = df_mol.groupby(["temperature", "pressure"]).agg(["mean", hmean])
    df_mol.columns = ["_".join(f) for f in df_mol.columns.tolist()]
    pandas.concat([df_mol, relaxations], axis=1).to_hdf(infile, "relaxations")


@sdanalysis.command()
def figure() -> None:
    """Start bokeh server with the file passed."""
    from bokeh.server.server import Server
    from bokeh.application import Application
    from bokeh.application.handlers.function import FunctionHandler

    from .figures.interactive_config import make_document

    apps = {"/": Application(FunctionHandler(make_document))}
    server = Server(apps)
    server.run_until_shutdown()
    logger.info("Bokeh server terminated.")
