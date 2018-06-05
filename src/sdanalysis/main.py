#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Run simulation with boilerplate taken care of by the statdyn library."""

import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Callable, List, Tuple

from ruamel.yaml import YAML

from .molecules import Dimer, Disc, Sphere, Trimer
from .params import SimulationParams
from .read import process_file
from .version import __version__

yaml = YAML()  # pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
logging.basicConfig(level="DEBUG")

gsd_logger = logging.getLogger("gsd")
gsd_logger.setLevel(logging.WARNING)

MOLECULE_OPTIONS = {"trimer": Trimer, "disc": Disc, "sphere": Sphere, "dimer": Dimer}


def sdanalysis() -> None:
    """Run main function."""
    logging.debug("Running main function")
    func, sim_params = parse_args()
    func(sim_params)


def comp_dynamics(sim_params: SimulationParams) -> None:
    """Compute dynamic properties."""
    outfile = sim_params.outfile_path / "dynamics.h5"
    outfile.parent.mkdir(exist_ok=True)
    sim_params.parameters["outfile"] = outfile
    try:
        with sim_params.mol_relaxations as src:
            sim_params.mol_relaxations = yaml.parse(src)
    except AttributeError:
        pass
    process_file(sim_params)


def figure(args) -> None:
    """Start bokeh server with the file passed."""
    from bokeh.server.server import Server
    from bokeh.application import Application
    from bokeh.application.handlers.function import FunctionHandler

    from .figures.interactive_config import make_document

    _verbosity(level=2)

    apps = {"/": Application(FunctionHandler(make_document))}
    server = Server(apps)
    server.run_until_shutdown()
    logger.info("Bokeh server terminated.")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        dest="outfile_path",
        type=str,
        help="Directory to output files to",
    )
    parse_molecule = parser.add_argument_group("molecule")
    parse_molecule.add_argument("--molecule", choices=MOLECULE_OPTIONS.keys())
    parse_molecule.add_argument(
        "--distance", type=float, help="Distance at which small particles are situated"
    )
    parse_molecule.add_argument(
        "--moment-inertia-scale",
        type=float,
        help="Scaling factor for the moment of inertia.",
    )
    parse_steps = parser.add_argument_group("steps")
    parse_steps.add_argument("--gen-steps", type=int)
    parse_steps.add_argument("--max-gen", type=int)
    default_parser = argparse.ArgumentParser(add_help=False)
    default_parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Enable debug logging flags."
    )
    default_parser.add_argument(
        "--version", action="version", version="sdanalysis {0}".format(__version__)
    )
    simtype = argparse.ArgumentParser(add_help=False, parents=[default_parser])
    subparsers = simtype.add_subparsers()
    parse_comp_dynamics = subparsers.add_parser(
        "comp_dynamics", add_help=False, parents=[parser, default_parser]
    )
    parse_comp_dynamics.add_argument("infile", type=str)
    parse_comp_dynamics.add_argument("-m", "--mol-relaxations", default=None, type=str)
    parse_comp_dynamics.set_defaults(func=comp_dynamics)
    parse_figure = subparsers.add_parser(
        "figure", add_help=True, parents=[default_parser]
    )
    parse_figure.add_argument("bokeh", nargs="*", default=[])
    parse_figure.set_defaults(func=figure)
    return simtype


def _verbosity(level: int = 1) -> None:
    root_logger = logging.getLogger("statdyn")
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    log_level = levels.get(level, logging.DEBUG)
    logging.basicConfig(level=log_level)
    root_logger.setLevel(log_level)


def parse_args(
    input_args: List[str] = None
) -> Tuple[Callable[[SimulationParams], None], SimulationParams]:
    """Logic to parse the input arguments."""
    parser = create_parser()
    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)
    # Handle verbosity
    _verbosity(args.verbose)
    del args.verbose
    # Handle subparser function
    try:
        func = args.func
        del args.func
    except AttributeError:
        parser.print_help()
        exit()
    # Parse Molecules
    my_mol = MOLECULE_OPTIONS.get(getattr(args, "molecule", None))
    if my_mol is None:
        my_mol = Trimer
    mol_kwargs = {}
    for attr in ["distance", "moment_inertia_scale"]:
        if getattr(args, attr, None) is not None:
            mol_kwargs[attr] = getattr(args, attr)
    args.molecule = my_mol(**mol_kwargs)
    set_args = {key: val for key, val in vars(args).items() if val is not None}
    return func, SimulationParams(**set_args)
