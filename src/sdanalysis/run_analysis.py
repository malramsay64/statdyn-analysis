#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Run an analysis on a trajectory."""

import logging

import gsd.hoomd
import numpy as np

from .order import orientational_order

logger = logging.getLogger(__name__)


def order(infile, outfile):
    """Compute the orientational order for each frame of a trajectory."""
    trajectory = gsd.hoomd.open(infile, "rb")
    with open(outfile, "w") as dst:
        print("Timestep, OrientOrder", file=dst)
        for index, _ in enumerate(trajectory):
            try:
                snapshot = trajectory[index]
            except RuntimeError:
                logger.info("Frame %s corrupted, continuing...", index)
                continue

            orient_order = orientational_order(
                box=snapshot.configuration.box,
                position=snapshot.particles.position,
                orientation=snapshot.particles.orientation,
            )
            print(
                snapshot.configuration.step,
                ",",
                np.sum(orient_order > 0.9) / len(orient_order),
                file=dst,
            )
