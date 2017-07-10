#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Run simulation with boilerplate taken care of by the statdyn library."""

from pathlib import Path
import logging

import click
import hoomd

from .. import crystals
from ..simulation import initialise, simrun

CRYSTAL_FUNCS = {
    'p2': crystals.TrimerP2,
}


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
    Path(value).mkdir(exist_ok=True)


def _verbosity(ctx, param, count):
    if not count or ctx.resilient_parsing:
        return
    if count:
        logging.basicConfig(level=logging.DEBUG)


@click.command()
@click.option('-c', '--configurations', type=click.Path(exists=True),
              required=True, help='location of configurations directory')
@click.option('-o', '--output',
              type=click.Path(file_okay=False, writable=True,),
              callback=_mkdir_ifempty, default='.')
@click.option('-v', '--verbose', count=True,
              expose_value=False, callback=_verbosity)
@click.option('--dynamics/--no-dynamics', default=True)
def main(configurations, output, dynamics):
    """Run main function."""
    configurations = Path(configurations)
    output = Path(output)
    kwargs = {}
    kwargs['output'] = output
    kwargs['configurations'] = configurations
    kwargs['dynamics'] = dynamics
    kwargs = {key: val for key, val in vars(args).items() if val is not None}
    if args.iterations > 1:
        simrun.iterate_random(**kwargs)
    else:
        simrun.run_npt(get_initial_snapshot(**kwargs), **kwargs)


if __name__ == "__main__":
    main()
