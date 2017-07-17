#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""sdrun commands for the equilibration of a configuration."""

import logging
from . import options, main
from ..simulation import initialise, equilibrate

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.WARNING)


@main.sdrun.group()
def equil():
    """Command group for the equilibration of configurations."""
    logger.info('Run equil')


@equil.command
@options.opt_temperature
@options.opt_steps
@options.opt_hoomd_args
@options.arg_infile
@options.arg_outfile
def interface(infile, outfile, temperature, steps, hoomd_args):
    """Equilibrate an interface."""
    snapshot = initialise.init_from_file(infile)
    equilibrate.equil_interface(
        snapshot,
        equil_temp=temperature,
        equil_steps=steps,
        hoomd_args=hoomd_args,
        outfile=outfile,
    )


@equil.command
@options.opt_temperature
@options.opt_steps
@options.opt_molecule
@options.opt_hoomd_args
@options.arg_infile
@options.arg_outfile
def liquid(infile, outfile, molecule, temperature, steps, hoomd_args):
    """Equilibrate a liquid."""
    snapshot = initialise.init_from_file(infile)
    equilibrate.equil_liquid(
        snapshot,
        equil_temp=temperature,
        equil_steps=steps,
        hoomd_args=hoomd_args,
        outfile=outfile,
    )
