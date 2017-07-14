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

from . import options
from .. import crystals
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
    # logging.debug(f"output: {output}")
    # configurations = Path(configurations)
    # kwargs = {}
    # kwargs['output'] = output
    # kwargs['configurations'] = configurations
    # kwargs['dynamics'] = dynamics
    # kwargs = {key: val for key, val in kwargs.items() if val is not None}
    # if args.iterations > 1:
    #    simrun.iterate_random(**kwargs)
    # else:
    #    simrun.run_npt(get_initial_snapshot(**kwargs), **kwargs)


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
        crystals.CRYSTAL_FUNCS.get(space_group)(),
        cell_dimensions=lattice_lengths
    )
    simrun.run_npt(
        snapshot,
        temp=temperature,
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
    snapshot = initialise.init_from_file(infile)
    logger.debug(f'Snapshot initialised')
    simrun.run_npt(
        snapshot,
        steps=steps,
        temp=temperature,
        dynamics=dynamics,
        output=output,
    )


if __name__ == "__main__":
    sdrun()
