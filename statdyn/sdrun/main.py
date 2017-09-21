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

import hoomd.context

from . import options
from ..simulation import equilibrate, initialise, simrun
from ..simulation.params import SimulationParams

logger = logging.getLogger(__name__)


def sdrun():
    """Run main function."""
    logging.debug('Running main function')


def prod(sim_params: SimulationParams) -> None:
    """Run simulations on equilibrated phase."""
    logger.debug('running prod')
    logger.debug('Reading %s', sim_params.infile)

    snapshot = initialise.init_from_file(sim_params.infile, hoomd_args=sim_params.hoomd_args)
    logger.debug(f'Snapshot initialised')

    sim_context = hoomd.context.initialize(sim_params.hoomd_args)
    simrun.run_npt(
        snapshot=snapshot,
        context=sim_context,
        sim_params=sim_params,
    )


def equil(sim_params: SimulationParams, equil_type: str) -> None:
    """Command group for the equilibration of configurations."""
    logger.debug('Running equil')

    # Ensure parent directory exists
    sim_params.outfile.parent.mkdir(exist_ok=True)

    snapshot = initialise.init_from_file(sim_params.infile)
    options.EQUIL_OPTIONS.get(equil_type)(
        snapshot,
        sim_params=sim_params,
    )


def create(sim_params: SimulationParams) -> None:
    """Create things."""
    logger.debug('Running create.')
    logger.debug('Interface flag: %s', sim_params.interface)
    # Ensure parent directory exists
    sim_params.outfile.parent.mkdir(exist_ok=True)

    snapshot = initialise.init_from_crystal(sim_params=sim_params)

    equilibrate.equil_crystal(
        snapshot=snapshot,
        sim_params=sim_params,
        interface=sim_params.interface,
    )


def figure(show_fig: str) -> None:
    """Start bokeh server with the file passed."""
    fig_file = Path(__file__).parents[1] / 'figures/interactive_config.py'
    try:
        run(['bokeh', 'serve', '--show', str(fig_file)])
    except ProcessLookupError:
        logger.info('Bokeh server terminated.')


if __name__ == "__main__":
    sdrun()
