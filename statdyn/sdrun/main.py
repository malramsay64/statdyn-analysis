#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Run simulation with boilerplate taken care of by the statdyn library."""

import logging

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
@options.opt_verbose
@options.opt_dynamics
@options.opt_hoomd_args
@options.arg_infile
def liquid(infile, steps, temperature, output, dynamics, hoomd_args):
    """Run simulations on liquid."""
    logger.debug(f'running liquid')
    logger.debug(f'Reading {infile}')
    snapshot = initialise.init_from_file(infile, hoomd_args=hoomd_args)
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
@options.opt_temperature
@options.opt_steps
@options.opt_hoomd_args
@options.opt_molecule
@options.opt_equil
@options.arg_infile
@options.arg_outfile
def equil(infile, outfile, molecule, temperature, steps, hoomd_args, equil_type):
    """Command group for the equilibration of configurations."""
    logger.info('Run equil')
    snapshot = initialise.init_from_file(infile)
    options.EQUIL_OPTIONS.get(equil_type)(
        snapshot,
        equil_temp=temperature,
        equil_steps=steps,
        hoomd_args=hoomd_args,
        molecule=molecule,
        outfile=outfile,
    )


@sdrun.command()
@options.opt_temperature
@options.opt_steps
@options.opt_hoomd_args
@options.opt_molecule
@options.opt_output
@options.opt_dynamics
@options.arg_infile
def dynamics(infile, temperature, molecule, steps, hoomd_args, output,
             dynamics):
    """Run simulation."""
    logger.info('Run equil')
    snapshot = initialise.init_from_file(infile)
    simrun.run_npt(snapshot,
                   context=hoomd.context.initialise(hoomd_args),
                   temperature=temperature,
                   steps=steps,
                   hoomd_args=hoomd_args,
                   mol=molecule
                   )


if __name__ == "__main__":
    sdrun()
