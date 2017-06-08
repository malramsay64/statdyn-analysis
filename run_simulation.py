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
import pandas
from statdyn import initialise, Simulation
import os


def get_fname(temp, ext='gsd'):
    return '{mol}-{press:.2f}-{temp:.2f}.{ext}'.format(
        mol='Trimer',
        press=13.50,
        temp=temp,
        ext=ext
    )

def get_temp(fname):
    from os.path import split, splitext
    fname = split(fname)[1]
    fname, ext = splitext(fname)
    return float(fname.split('-')[2])


def get_closest(temp, directory):
    files = glob.glob(directory+"/*.gsd")
    return temp+min([get_temp(t) - temp for t in files if get_temp(t) > temp])


def main(steps, temp, directory, output, thermo=True):
    if glob.glob(directory+'/'+get_fname(temp)):
        snapshot = initialise.init_from_file(
            directory+'/'+get_fname(temp)).take_snapshot()
    else:
        snapshot = initialise.init_from_file(directory+'/'+get_fname(
            get_closest(temp, directory))).take_snapshot()
    data = Simulation.run_multiple_concurrent(
        snapshot,
        temp,
        steps,
        thermo=thermo
    )
    os.makedirs(output, exist_ok=True)
    with pandas.HDFStore(output+'/'+get_fname(temp, 'hdf5')) as dst:
        dst['dynamics'] = data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--steps',
        type=int,
        help='The number of steps to run the simulation for'
    )
    parser.add_argument(
        '-t',
        '--temp',
        type=float,
        help='The temperature of the simulation'
    )
    parser.add_argument(
        '-d',
        '--dir',
        type=str,
        help='The directory in which to find the gsd configuration files'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='The directory to store the log files and data files'
    )
    parser.add_argument(
        '--no-thermo',
        action='store_false',
        help='Do not output thermodynamic data to a file'
    )
    args = parser.parse_args()
    main(args.steps, args.temp, args.dir, args.output)

