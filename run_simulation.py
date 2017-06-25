#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Run simulation with boilerplate taken care of by the statdyn library."""

import argparse
from pathlib import Path

import hoomd
from statdyn import Simulation, crystals, initialise

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
    Simulation.run_npt(snapshot, 0.1, 1)


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


def main():
    """Run main function."""
    args = _argument_parser().parse_args()
    args.dir = Path(args.dir)
    args.output = Path(args.output)
    args.output.mkdir(exist_ok=True)
    kwargs = {key: val for key, val in vars(args).items() if val is not None}
    if args.iterations > 1:
        Simulation.iterate_random(**kwargs)
    else:
        Simulation.run_npt(get_initial_snapshot(**kwargs), **kwargs)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--steps',
        required=True,
        type=int,
        help='The number of steps to run the simulation for'
    )
    parser.add_argument(
        '-t',
        '--temp',
        type=float,
        required=True,
        help='The temperature of the simulation'
    )
    parser.add_argument(
        '-d',
        '--dir',
        type=str,
        default='.',
        help='The directory in which to find the gsd configuration files'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='.',
        help='The directory to store the log files and data files'
    )
    parser.add_argument(
        '--no-thermo',
        dest='thermo',
        action='store_false',
        help='Do not output thermodynamic data to a file'
    )
    parser.add_argument(
        '--thermo',
        dest='thermo',
        action='store_true',
        help='Output thermodynamics data to a file (defualt)'
    )
    parser.add_argument(
        '--init-crys',
        default=None,
        help='Initialise the simulation from a crystal'
    )
    parser.add_argument(
        '--lattice-lengths',
        nargs=2,
        default=(30, 40),
        help='Number of repetitions in the a and b lattice vectors'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=1,
        help='Number of random iterations to run'
    )
    parser.add_argument(
        '--hoomd-args',
        dest='init_args',
        type=str,
        default='',
        help='String of arguments to pass to hoomd on initialisation'
    )
    parser.add_argument(
        '--dyn-many',
        dest='dyn_many',
        action='store_true',
        help='''Have many starting configurations for dynamics in a single
        simulation'''
    )
    parser.add_argument(
        '--dyn-single',
        dest='dyn_many',
        action='store_false',
        help='Have only a single starting configuration for dynamics'
    )
    parser.add_argument(
        '--thermo-period',
        help='Period thermodynamic properties are dumped'
    )
    parser.add_argument(
        '--dump-period',
        help='Period trajectory is dumped'
    )
    parser.set_defaults(thermo=True)
    parser.set_defaults(dyn_many=True)
    return parser


if __name__ == "__main__":
    main()
