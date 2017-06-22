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

import pandas
from statdyn import Simulation, crystals, initialise

CRYSTAL_FUNCS = {
    'p2': crystals.TrimerP2,
}


def get_temp(fname: Path) -> float:
    """Convert a filename to a temperature."""
    return float(fname.stem.split('-')[2])


def get_closest(temp: float, directory: Path) -> float:
    """Find the closet already equilibrated temperature."""
    files = directory.glob('/*.gsd')
    return temp + min(
        [get_temp(t) - temp for t in files if get_temp(t) > temp])


def crystalline():
    """Run a crystalline simulation."""
    snapshot = initialise.init_from_crystal(crystals.TrimerP2).take_snapshot()
    Simulation.run_npt(snapshot, 0.1, 1)


def main():
    """Run main function."""
    args = _argument_parser().parse_args()
    args.dir = Path(args.dir)
    args.output = Path(args.output)
    args.output.mkdir(exist_ok=True)
    if args.init_crys:
        crys = CRYSTAL_FUNCS.get(args.init_crys)
        snapshot = initialise.init_from_crystal(
            crys(),
            cell_dimensions=args.lattice_lengths,
            init_args=args.hoomd_args,
        ).take_snapshot()
    elif (args.dir / initialise.get_fname(args.temp)).is_file():
        snapshot = initialise.init_from_file(
            args.dir / initialise.get_fname(args.temp),
            init_args=args.hoomd_args,
        ).take_snapshot()
    else:
        snapshot = initialise.init_from_file(
            args.dir / initialise.get_fname(get_closest(args.temp, args.dir))
        ).take_snapshot()
    if args.iterations:
        Simulation.iterate_random(
            args.dir,
            args.temp,
            args.steps,
            args.iterations,
            args.output,
            init_args=args.hoomd_args,
        )
    else:
        data = Simulation.run_npt(
            snapshot,
            args.temp,
            args.steps,
            thermo=args.thermo,
            init_args=args.hoomd_args,
            output=args.output,
        )
        outfile = args.output / initialise.get_fname(args.temp, 'hdf5')
        with pandas.HDFStore(outfile) as dst:
            dst['dynamics'] = data


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
        default=None,
        help='Number of random iterations to run'
    )
    parser.add_argument(
        '--hoomd-args',
        type=str,
        default='',
        help='String of arguments to pass to hoomd on initialisation'
    )
    parser.set_defaults(thermo=True)
    return parser


if __name__ == "__main__":
    main()
