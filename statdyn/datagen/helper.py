#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Helper functions for figure generation.

This is a series of helper functions for the generation of figures that
are representative of the dynamic properties of a molecular dynamics system.
"""

from pathlib import Path

import pandas

TEMP_INDEX = 2


def read_hdf5(filename: Path) -> pandas.DataFrame:
    """Read a series of translations and rotations to a pandas DataFrame.

    Args:
        filename (class:`pathlib.Path`): This is the filename of the actual
            file from which to read.

    Returns:
        class:`pandas.DataFrame`: A DataFrame containing the translational and
            rotational motion of each particle at a number of timesteps.

    """
    with pandas.HDFStore(filename, 'r') as src:
        data = src
    return data


def read_dat(directory: Path) -> pandas.DataFrame:
    """Read a series of data files to a pandas DataFrame.

    This looks for all the dynamcis files in a directory, returning a DataFrame
    containing all the data from the files.

    Args:
        directory (class:`pathlib.Path`): Directory in which to serach for
            files

    Returns:
        class:`pandas.DataFrame`: All the data from the files in the directory
            provided to the function.

    """
    files = directory.glob('*-dyn.dat')
    data = {}
    for src in files:
        data[get_temperature(src)] = pandas.read_table(src, sep=' ')
    return pandas.DataFrame(data)


def get_temperature(fname: Path) -> str:
    """Convert a filename to temperature.

    This takes a filename and returns the temperature at which the simulation
    was run based on my naming scheme.

    Args:
        fname (class:`pathlib.Path`): The filename to get the temperature from

    Returns:
        str: Temperature of the simulation

    """
    return fname.stem.split('-')[TEMP_INDEX]
