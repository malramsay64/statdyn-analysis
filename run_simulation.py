#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Run simulation with boilerplate taken care of by the statdyn library
"""

import argparse
import glob
import os
import pandas
from statdyn import initialise, Simulation, crystals


crystal_funcs={
    'p2': crystals.p2,
}

def get_temp(fname):
    from os.path import split, splitext
    fname = split(fname)[1]
    fname, ext = splitext(fname)
    return float(fname.split('-')[2])


def get_closest(temp, directory):
    files = glob.glob(directory+"/*.gsd")
    return temp+min([get_temp(t) - temp for t in files if get_temp(t) > temp])


def crystalline():
    snapshot = initialise.init_from_crystal(crystals.p2()).take_snapshot()
    Simulation.run_npt(snapshot, 0.1, 1)


def main(steps, temp, directory, output, thermo=True, init_crys=None, lattice_lengths=(30, 40)):
    if init_crys:
        crys = crystal_funcs.get(init_crys)
        snapshot = initialise.init_from_crystal(crys(), cell_dimensions=lattice_lengths).take_snapshot()
    elif glob.glob(directory+'/'+initialise.get_fname(temp)):
        snapshot = initialise.init_from_file(
            directory+'/'+initialise.get_fname(temp)).take_snapshot()
    else:
        snapshot = initialise.init_from_file(directory+'/'+initialise.get_fname(
            get_closest(temp, directory))).take_snapshot()
    data = Simulation.run_npt(
        snapshot,
        temp,
        steps,
        thermo=thermo
    )
    os.makedirs(output, exist_ok=True)
    with pandas.HDFStore(output+'/'+initialise.get_fname(temp, 'hdf5')) as dst:
        dst['dynamics'] = data


if __name__ == "__main__":
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
    parser.set_defaults(thermo=True)
    args = parser.parse_args()
    main(args.steps, args.temp, args.dir, args.output, args.thermo, args.init_crys, args.lattice_lengths)

