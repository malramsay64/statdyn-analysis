#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""CLI module for the creation of initial configurations."""


import logging
import click

from .main import sdrun
from ..simulation import equilibrate
import options

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.WARNING)


@sdrun.group(name='create')
def create():
    """Group for the creation of stuff."""
    logger.info('Running create')


@create.command()
@options.opt_space_group
@options.opt_lattice_lengths
@options.hoomd_args
@click.argument('outfile', type=click.File('wb'))
def crystal(space_group, lattice_lengths, hoomd_args, outfile):
    """Generate a crystal configuration."""
    initialise.init_from_crystal(
        crystal=space_group,
        hoomd_args=hoomd_args,
        cell_dimensions=lattice_lengths,
        outfile=Path(outfile),
    )


@create.command()
@options.opt_space_group
@options.opt_lattice_lengths
@options.opt_temperature
@options.opt_steps
@options.opt_hoomd_args
@click.argument('outfile', type=click.File('wb'))
def interface(space_group, lattice_lengths, temperature,
              steps, outfile, hoomd_args):
    """TODO: Docstring for interface.

    Args:
        space_group (TODO): TODO
        lattice_lengths (TODO): TODO
        hoomd_args (TODO): TODO
        outfile (TODO): TODO
        temperature (TODO): TODO
        steps (TODO): TODO

    Returns: TODO

    """
    snapshot = initialise.init_from_crystal(
        crystal=space_group,
        hoomd_args=hoomd_args,
        cell_dimensions=lattice_lengths,
    )
    equilibrate.create_interface(
        snapshot=snapshot,
        melt_temp=temperature,
        melt_steps=steps,
        outfile=Path(outfile),
    )
