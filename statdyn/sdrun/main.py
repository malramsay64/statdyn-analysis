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

import click
import hoomd.context

from . import options
from ..simulation import initialise, simrun

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)


@click.group(name='sdrun')
@click.version_option()
@click.pass_context
def sdrun(ctx):
    """Run main function."""
    logging.debug('Running main function')


@sdrun.command()
@options.opt_space_group
@options.opt_lattice_lengths
@options.opt_steps
@options.opt_temperature
@options.opt_dynamics
@options.opt_output
@options.opt_verbose
@options.opt_hoomd_args
def crystal(space_group, lattice_lengths, steps,
            temperature, dynamics, output, hoomd_args):
    """Run simulations on crystals."""
    snapshot = initialise.init_from_crystal(
        crystal=space_group,
        hoomd_args=hoomd_args,
        cell_dimensions=lattice_lengths
    )
    sim_context = hoomd.context.initialize(hoomd_args)
    simrun.run_npt(
        snapshot=snapshot,
        context=sim_context,
        temperature=temperature,
        steps=steps,
        dynamics=dynamics,
        output=output,
    )


@sdrun.command()
@options.opt_steps
@options.opt_temperature
@options.opt_output
@options.opt_configurations
@options.opt_verbose
@options.opt_dynamics
@options.opt_hoomd_args
def liquid(steps, temperature, output, configurations, dynamics, hoomd_args):
    """Run simulations on liquid."""
    logger.debug(f'running liquid')
    infile = Path(configurations) / initialise.get_fname(temperature)
    logger.debug(f'Reading {infile}')
    snapshot = initialise.init_from_file(infile,
                                         hoomd_args=hoomd_args)
    logger.debug(f'Snapshot initialised')
    sim_context = hoomd.context.initialize(hoomd_args)
    simrun.run_npt(
        snapshot=snapshot,
        context=sim_context,
        steps=steps,
        temperature=temperature,
        dynamics=dynamics,
        output=output,
    )


@sdrun.command()
@options.opt_steps
@options.opt_temperature
@options.opt_space_group
@options.opt_lattice_lengths
@options.opt_output
@options.opt_verbose
@options.opt_hoomd_args
def create_slab(steps, temperature, space_group, lattice_lengths, output,
                hoomd_args):
    """Create a slab of crystal in a liquid."""
    logger.debug('Running create_slab')
    initialise.init_slab(
        crystal=space_group,
        hoomd_args=hoomd_args,
        equil_temp=temperature,
        equil_steps=steps,
        melt_temp=1.5*temperature,
        melt_steps=steps,
        cell_dimensions=lattice_lengths,
    )


if __name__ == "__main__":
    sdrun()
