#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Options for sdrun."""

import argparse
import logging

from ..analysis.run_analysis import comp_dynamics
from ..crystals import CRYSTAL_FUNCS
from ..molecules import Dimer, Disc, Sphere, Trimer
from ..simulation import equilibrate
from ..simulation.helper import SimulationParams
from .main import create, equil, figure, prod

logger = logging.getLogger(__name__)

EQUIL_OPTIONS = {
    'interface': equilibrate.equil_interface,
    'liquid': equilibrate.equil_liquid,
    'crystal': equilibrate.equil_crystal,
}

MOLECULE_OPTIONS = {
    'trimer': Trimer,
    'disc': Disc,
    'sphere': Sphere,
    'dimer': Dimer,
}


def _verbosity(ctx, param, count):
    root_logger = logging.getLogger('statdyn')
    if not count or ctx.resilient_parsing:
        logging.basicConfig(level=logging.WARNING)
        root_logger.setLevel(logging.WARNING)
        return
    if count == 1:
        logging.basicConfig(level=logging.INFO)
        root_logger.setLevel(logging.INFO)
        logger.info('Set log level to INFO')
    if count > 1:
        logging.basicConfig(level=logging.DEBUG)
        root_logger.setLevel(logging.DEBUG)
        logger.info('Setting log level to DEBUG')
    root_logger.debug('Logging set for root')


def parse_args():
    """Create the argument parser."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
        help='Enable debug logging flags.',
    )
    parser.add_argument(
        '-s', '--steps',
        dest='num_steps',
        type=int,
        help='The number of steps for which to run the simulation.',
    )
    parser.add_argument(
        '--moment-inertia-scale',
        type=float,
        help='Scaling factor for the moment of inertia.',
    )
    parser.add_argument(
        '--output-interval',
        type=int,
        help='Steps between output of dump and thermodynamic quantities.'
    )
    parser.add_argument(
        '--hoomd-args',
        type=str,
        help='Arguments to pass to hoomd on context.initialize',
    )
    parser.add_argument(
        '--pressure',
        type=float,
        help='Pressure for simulation',
    )
    parser.add_argument(
        '-t',
        '--temperature',
        type=float,
        help='Temperature for simulation',
    )
    parser.add_argument(
        '-i',
        '--output',
        type=str,
        help='Directory to output files to',
    )
    parser.add_argument('outfile', type=str)
    parser.add_argument('infile', type=str)

    parse_molecule = parser.add_argument_group('molecule')
    parse_molecule.add_argument(
        '--molecule',
        choices=MOLECULE_OPTIONS.keys(),
    )

    parse_crystal = parser.add_argument_group('crystal')
    parse_crystal.add_argument(
        '--space-group',
        choices=CRYSTAL_FUNCS.keys(),
        help='Space group of initial crystal.',
    )
    parse_crystal.add_argument(
        '--lattice-lengths',
        nargs=2,
        default=(30, 42),
        type=int,
        help='Number of repetitiions in the a and b lattice vectors',
    )

    # TODO write up something useful in the help
    subparsers = parser.add_subparsers()
    equilibration = subparsers.add_parser('equilibration', add_help=False, parents=[parser])
    equilibration.add_argument(
        '--init-temp',
        type=float,
        help='Temperature to start equilibration from if differnt from the target.'
    )
    equilibration.add_argument(
        '--equil-type',
        default='liquid',
        choices=EQUIL_OPTIONS.keys(),
    )
    equilibration.set_defaults(func=equil)

    production = subparsers.add_parser('prod', add_help=False, parents=[parser])
    production.add_argument('--dynamics', action='store_true')
    production.add_argument('--no-dynamics', action='store_false')
    production.set_defaults(func=prod)

    parse_comp_dynamics = subparsers.add_parser('comp_dynamics', add_help=False, parents=[parser])
    parse_comp_dynamics.set_defaults(func=comp_dynamics)
# @click.option('--gen-steps', default=20000, type=click.IntRange(min=0))
# @click.option('--max-gen', default=500, type=click.IntRange(min=1))
# @click.option('--step-limit', default=None, type=click.IntRange(min=1))

    parse_create = subparsers.add_parser('create', add_help=False, parents=[parser])
    parse_create.set_defaults(func=create)
    parse_figure = subparsers.add_parser('figure', add_help=True)
    parse_figure.set_defaults(func=figure)


def create_params(**kwargs) -> SimulationParams:
    """Create the input parameters from arguments."""
    return SimulationParams(**kwargs)
