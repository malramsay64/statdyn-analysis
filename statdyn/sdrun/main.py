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
from ..simulation import equilibrate, initialise, simrun

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.WARN)


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
    # Create crystal
    snapshot = initialise.init_from_crystal(
        crystal=space_group,
        hoomd_args=hoomd_args,
        cell_dimensions=lattice_lengths
    )
    molecule = space_group.molecule

    # Equilibrate Crystal
    equilibrate.equil_crystal(
        snapshot,
        equil_temp=temperature,
        equil_steps=steps,
        hoomd_args=hoomd_args,
        molecule=molecule,
    )

    # Run simulation
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
@options.opt_verbose
@options.opt_steps
@options.opt_temperature
@options.opt_output
@options.opt_molecule
@options.opt_verbose
@options.opt_dynamics
@options.opt_hoomd_args
@options.arg_infile
def prod(infile, steps, temperature, molecule, output,
         dynamics, hoomd_args):
    """Run simulations on equilibrated phase."""
    logger.debug(f'running prod')
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
@options.opt_init_temp
@options.arg_infile
@options.arg_outfile
def equil(infile, outfile, molecule, temperature, steps,
          init_temp, hoomd_args, equil_type):
    """Command group for the equilibration of configurations."""
    logger.info('Run equil')

    # Ensure parent directory exists
    Path(outfile).parent.mkdir(exist_ok=True)

    snapshot = initialise.init_from_file(infile)
    options.EQUIL_OPTIONS.get(equil_type)(
        snapshot,
        equil_temp=temperature,
        equil_steps=steps,
        hoomd_args=hoomd_args,
        molecule=molecule,
        init_temp=init_temp,
        outfile=outfile,
    )


@sdrun.command()
@options.opt_verbose
@options.opt_space_group
@options.opt_lattice_lengths
@options.opt_temperature
@options.opt_steps
@options.opt_hoomd_args
@options.arg_outfile
@click.option('--interface', is_flag=True)
def create(space_group, lattice_lengths, temperature, steps,
           outfile, interface, hoomd_args):
    """Create things."""
    # Ensure parent directory exists
    Path(outfile).parent.mkdir(exist_ok=True)

    snapshot = initialise.init_from_crystal(
        crystal=space_group,
        hoomd_args=hoomd_args,
        cell_dimensions=lattice_lengths,
        outfile=None,
    )

    equilibrate.equil_crystal(
        snapshot=snapshot,
        equil_temp=temperature,
        equil_steps=steps,
        outfile=outfile,
        interface=interface
    )


@sdrun.command()
@options.opt_verbose
@options.opt_temperature
@options.opt_steps
@options.opt_molecule
@options.opt_output
@options.opt_hoomd_args
@options.opt_dynamics
@options.opt_init_temp
@options.arg_infile
def interface(infile, temperature, steps, dynamics, molecule,
              init_temp, output, hoomd_args):
    """Create things."""
    # Initialise
    snapshot = initialise.init_from_file(infile, hoomd_args=hoomd_args)

    if init_temp is None:
        init_temp = 4.00
    # Equilibrate
    snapshot = equilibrate.equil_interface(
        snapshot=snapshot,
        equil_temp=temperature,
        equil_steps=steps,
        init_temp=init_temp,
    )

    # Production
    context = hoomd.context.initialize(hoomd_args)
    simrun.run_npt(
        snapshot,
        context=context,
        output=output,
        steps=steps,
        temperature=temperature,
        dynamics=dynamics,
    )


if __name__ == "__main__":
    sdrun()
