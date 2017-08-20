#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Options for sdrun."""

import logging
from pathlib import Path

import click

from ..crystals import CRYSTAL_FUNCS
from ..molecule import Trimer
from ..simulation import equilibrate

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

EQUIL_OPTIONS = {
    'interface': equilibrate.equil_interface,
    'liquid': equilibrate.equil_liquid
}


def _mkdir_ifempty(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    chkpath = Path(value)
    logging.debug(f"Directory {value}, {chkpath}")
    chkpath.mkdir(exist_ok=True)
    return chkpath


def _verbosity(ctx, param, count):
    if not count or ctx.resilient_parsing:
        return
    logger.info('Setting log level to DEBUG')
    # logging.basicConfig(level=logging.DEBUG)


def _create_crystal(ctx, param, crys):
    if not crys or ctx.resilient_parsing:
        return
    return CRYSTAL_FUNCS.get(crys)()


def _get_molecule(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    return Trimer()


opt_space_group = click.option(
    '--space-group',
    default='p2',
    type=click.Choice(CRYSTAL_FUNCS.keys()),
    callback=_create_crystal,
    help='Space group of initial crystal.',
)


opt_lattice_lengths = click.option(
    '--lattice-lengths',
    nargs=2,
    default=(30, 40),
    type=click.Tuple([int, int]),
    help='Number of repetitiions in the a and b lattice vectors',
)


opt_configurations = click.option(
    '-c',
    '--configurations',
    default='.',
    type=click.Path(exists=True),
    help='location of configurations directory',
)


opt_output = click.option(
    '-o',
    '--output',
    default='.',
    type=click.Path(file_okay=False, writable=True,),
    callback=_mkdir_ifempty,
    help='Directory to which output files will be written.',
)


opt_verbose = click.option(
    '-v',
    '--verbose',
    count=True,
    expose_value=False,
    callback=_verbosity,
    help='Enable debug logging flags.',
)


opt_dynamics = click.option(
    '--dynamics/--no-dynamics',
    default=True,
    help='''Enable/diable the collection of dynamics quantities.

Enabling the dynamics quantities will collect in addition to the standard dump
file another trajectory in which the configurations are saved on a logarithmic
scale.
'''
)


opt_steps = click.option(
    '-s',
    '--steps',
    type=click.IntRange(min=0, max=int(1e12)),
    help='Number of steps to run simulation for.'
)


opt_temperature = click.option(
    '-t',
    '--temperature',
    type=float,
    help='Temperature for simulation',
)


opt_hoomd_args = click.option(
    '--hoomd-args',
    type=str,
    default='',
    help='Arguments to pass to hoomd on context.initialize',
)

opt_molecule = click.option(
    '--molecule',
    default='trimer',
    type=click.Choice(['trimer']),
    callback=_get_molecule
)

opt_equil = click.option(
    '--equil-type',
    default='liquid',
    type=click.Choice(EQUIL_OPTIONS.keys()),
)

opt_init_temp = click.option(
    '--init-temp',
    default=None,
    type=float,
    help='Temperature to start equilibration from if differnt from the target.'
)

opt_output_interval = click.option(
    '--output-interval',
    default=10000,
    type=click.IntRange(min=0),
    help='Steps between output of dump and thermodynamic quantities.'
)

arg_infile = click.argument(
    'infile',
    type=click.Path(exists=True, dir_okay=False, readable=True),
)

arg_outfile = click.argument(
    'outfile',
    type=click.Path(exists=False, dir_okay=False, writable=True),
)
