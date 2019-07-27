#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Module for reading and processing input files."""


from ._gsd import read_gsd_trajectory
from ._lammps import read_lammps_trajectory
from ._read import open_trajectory, process_file
