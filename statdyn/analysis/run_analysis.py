#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Run an analysis on a trajectory."""

import logging
from pathlib import Path

import click
import gsd.hoomd
import numpy as np

from ..sdrun import options
from .order import orientational_order
from .read import process_gsd

logger = logging.getLogger(__name__)


@click.command()
@options.arg_infile
@options.arg_outfile
def order(infile, outfile):
    """Compute the orientational order for each frame of a trajectory."""
    trajectory = gsd.hoomd.open(infile, 'rb')
    with open(outfile, 'w') as dst:
        print('Timestep, OrientOrder', file=dst)
        for index in range(len(trajectory)):
            try:
                snapshot = trajectory[index]
            except RuntimeError:
                logger.info('Frame %s corrupted, continuing...', index)
                continue
            order = orientational_order(snapshot)
            print(f'{snapshot.configuration.step}, {np.sum(order > 0.9) / len(order)}',
                  file=dst)


@click.command()
@options.arg_infile
@options.opt_output
@options.opt_verbose
@click.option('--gen-steps', default=20000, type=click.IntRange(min=0))
@click.option('--max-gen', default=500, type=click.IntRange(min=1))
@click.option('--step-limit', default=None, type=click.IntRange(min=1))
def comp_dynamics(infile, output, gen_steps, step_limit, max_gen):
    """Compute dynamic properties."""
    outfile = str(output / Path(infile).with_suffix('.hdf5').name)
    dynamics_data = process_gsd(infile,
                                gen_steps=gen_steps,
                                max_gen=max_gen,
                                step_limit=step_limit,
                                outfile=outfile,
                                )
