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
import hoomd

from .. import crystals
from ..simulation import initialise, simrun

logger = logging.getLogger(__name__)


def get_temp(fname: Path) -> float:
    """Convert a filename to a temperature."""
    return float(fname.stem.split('-')[2])


def get_closest(temp: float, directory: Path) -> float:
    """Find the closet already equilibrated temperature."""
    files = directory.glob('*.gsd')
    return temp + min(
        [get_temp(t) - temp for t in files if get_temp(t) > temp])


def crystalline():
    """Run a crystalline simulation."""
    snapshot = initialise.init_from_crystal(crystals.TrimerP2).take_snapshot()
    simrun.run_npt(snapshot, 0.1, 1)


def get_initial_snapshot(**kwargs) -> hoomd.data.SnapshotParticleData:
    """Create the appropriate initial snapshot based on kwargs."""
    if kwargs.get('init_crys'):
        crys = CRYSTAL_FUNCS.get(kwargs.get('init_crys'))
        return initialise.init_from_crystal(
            crys(),
            cell_dimensions=kwargs.get('lattice_lengths'),
            init_args=kwargs.get('init_args'),
        ).take_snapshot()

    infile = kwargs.get('dir') / initialise.get_fname(kwargs.get('temp'))
    if infile.is_file():
        return initialise.init_from_file(
            infile,
            init_args=kwargs.get('init_args'),
        ).take_snapshot()
    else:
        return initialise.init_from_file(
            kwargs.get('dir') / initialise.get_fname(
                get_closest(kwargs.get('temp'), kwargs.get('dir')))
        ).take_snapshot()


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
    if count:
        logging.basicConfig(level=logging.DEBUG)


opt_space_group = click.option(
    '--space-group',
    default='p2',
    type=click.Choice(crystals.CRYSTAL_FUNCS.keys())
)


opt_lattice_lengths = click.option(
    '--lattice-lengths',
    nargs=2,
    default=(30, 40),
    type=(int, int),
    help='Number of repetitiions in the a and b lattice vectors'
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
    type=click.Path(file_okay=False, writable=True,),
    callback=_mkdir_ifempty,
    default='.'
)


opt_verbose = click.option(
    '-v',
    '--verbose',
    count=True,
    expose_value=False,
    callback=_verbosity
)


opt_dynamics = click.option(
    '--dynamics/--no-dynamics',
    default=True
)


opt_steps = click.option(
    '-s',
    '--steps',
    type=click.IntRange(min=0, max=1e12),
    required=True,
    help='Number of steps to run simulation for'
)


opt_temperature = click.option(
    '-t',
    '--temperature',
    type=float,
    required=True,
    help='Temperature for simulation'
)


@click.group(name='sdrun')
@click.pass_context
def main(ctx):
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


@main.command()
@opt_space_group
@opt_lattice_lengths
@opt_steps
@opt_temperature
@opt_dynamics
@opt_output
@opt_verbose
@click.pass_context
def crystal(ctx, space_group, lattice_lengths, steps,
            temperature, dynamics, output):
    """Run simulations on crystals."""
    snapshot = initialise.init_from_crystal(
        crystals.CRYSTAL_FUNCS.get(space_group)(),
        cell_dimensions=lattice_lengths
    ).take_snapshot()
    simrun.run_npt(
        snapshot,
        temp=temperature,
        steps=steps,
        dynamics=dynamics,
        output=output,
    )


if __name__ == "__main__":
    main()
