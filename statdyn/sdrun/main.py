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
from subprocess import run
from typing import Tuple

import click
import hoomd.context

from . import options
from ..analysis.run_analysis import order
from ..crystals import Crystal
from ..datagen.summary import dynamics, motion
from ..molecule import Molecule
from ..simulation import equilibrate, initialise, simrun

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@click.group(name='sdrun')
@click.version_option()
@options.opt_verbose
@click.pass_context
def sdrun(ctx):
    """Run main function."""
    # logging.debug('Running main function')


@sdrun.command()
@options.opt_verbose
@options.opt_steps
@options.opt_temperature
@options.opt_output
@options.opt_molecule
@options.opt_verbose
@options.opt_dynamics
@options.opt_hoomd_args
@options.opt_output_interval
@options.arg_infile
def prod(infile: str,
         steps: int,
         temperature: float,
         molecule: Molecule,
         output: str,
         dynamics: bool,
         hoomd_args: str,
         output_interval: int,
         ) -> None:
    """Run simulations on equilibrated phase."""
    logger.debug(f'running prod')
    logger.debug(f'Reading {infile}')

    snapshot = initialise.init_from_file(Path(infile), hoomd_args=hoomd_args)
    logger.debug(f'Snapshot initialised')

    sim_context = hoomd.context.initialize(hoomd_args)
    simrun.run_npt(
        snapshot=snapshot,
        context=sim_context,
        steps=steps,
        temperature=temperature,
        dynamics=dynamics,
        output=Path(output),
        dump_period=output_interval,
        thermo_period=output_interval,
    )


@sdrun.command()
@options.opt_verbose
@options.opt_temperature
@options.opt_steps
@options.opt_hoomd_args
@options.opt_molecule
@options.opt_equil
@options.opt_init_temp
@options.arg_infile
@options.arg_outfile
def equil(infile: str,
          outfile: str,
          molecule: Molecule,
          temperature: float,
          steps: int,
          init_temp: float,
          hoomd_args: str,
          equil_type: str,
          ) -> None:
    """Command group for the equilibration of configurations."""
    logger.info('Run equil')

    # Ensure parent directory exists
    outfile_path = Path(outfile)
    outfile_path.parent.mkdir(exist_ok=True)

    snapshot = initialise.init_from_file(Path(infile))
    options.EQUIL_OPTIONS.get(equil_type)(
        snapshot,
        equil_temp=temperature,
        equil_steps=steps,
        hoomd_args=hoomd_args,
        molecule=molecule,
        init_temp=init_temp,
        outfile=outfile_path,
    )


@sdrun.command()
@options.opt_verbose
@options.opt_space_group
@options.opt_lattice_lengths
@options.opt_molecule
@options.opt_temperature
@options.opt_steps
@options.opt_hoomd_args
@options.arg_outfile
@click.option('--interface', is_flag=True)
def create(space_group: Crystal,
           lattice_lengths: Tuple[int, int],
           molecule: Molecule,
           temperature: float,
           steps: int,
           outfile: str,
           interface: bool,
           hoomd_args: str,
           ) -> None:
    """Create things."""
    logger.debug('Interface flag: {interface}')
    outfile_path = Path(outfile)
    # Ensure parent directory exists
    outfile_path.parent.mkdir(exist_ok=True)

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
        outfile=Path(outfile),
        interface=interface,
        molecule=molecule,
    )


@sdrun.command()
@options.opt_verbose
@click.option('--show-fig', type=click.Choice(['interactive']),
              default='interactive')
def figure(show_fig: str) -> None:
    """Start bokeh server with the file passed."""
    lookup = {
        'interactive': Path(__file__).parents[1] / 'figures/interactive_config.py',
    }
    try:
        run(['bokeh', 'serve', '--show', str(lookup.get(show_fig))])
    except ProcessLookupError:
        logger.info('Bokeh server terminated.')


for ext_command in [order, dynamics, motion]:
    sdrun.add_command(ext_command)

if __name__ == "__main__":
    sdrun()
