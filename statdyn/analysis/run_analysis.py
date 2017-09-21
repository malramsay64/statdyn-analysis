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

import gsd.hoomd
import numpy as np

from .order import orientational_order
from .read import process_gsd

logger = logging.getLogger(__name__)


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
            order = orientational_order(
                box=snapshot.configuration.box,
                position=snapshot.particles.position,
                orientation=snapshot.particles.orientation
            )
            print(f'{snapshot.configuration.step}, {np.sum(order > 0.9) / len(order)}',
                  file=dst)


def comp_dynamics(infile, output, gen_steps, step_limit, max_gen):
    """Compute dynamic properties."""
    outfile = str(output / Path(infile).with_suffix('.hdf5').name)
    process_gsd(infile,
                gen_steps=gen_steps,
                max_gen=max_gen,
                step_limit=step_limit,
                outfile=outfile,
                )
