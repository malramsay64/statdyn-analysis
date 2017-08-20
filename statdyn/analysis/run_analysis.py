#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Run an analysis on a trajectory."""

import click
import gsd.hoomd
import numpy as np

from ..sdrun import options
from .order import orientational_order


@click.command()
@options.arg_infile
@options.arg_outfile
def order(infile, outfile):
    """Compute the orientational order for each frame of a trajectory."""
    trajectory = gsd.hoomd.open(infile, 'rb')
    with open(outfile, 'w') as dst:
        print('Timestep, OrientOrder', file=dst)
        for snapshot in trajectory:
            order = orientational_order(snapshot)
            print(f'{snapshot.configuration.step}, {np.sum(order > 0.9) / len(order)}',
                  file=dst)
