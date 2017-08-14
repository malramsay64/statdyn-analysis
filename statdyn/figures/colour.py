#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Create functions to colourize figures."""

import logging

import numpy as np
import quaternion
from hsluv import hpluv_to_hex

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

HEX_VALUES = np.array([hpluv_to_hex((value, 85, 65)) for value in range(360)])


def clean_orientation(snapshot):
    """Convert an orientation to a sensible format."""
    orientation = quaternion.as_rotation_vector(
        quaternion.as_quat_array(snapshot.particles.orientation.astype(float))).sum(axis=1)
    orientation[orientation <= -np.pi] += 2*np.pi
    orientation[orientation > np.pi] -= 2*np.pi
    return orientation
    nmol = max(snapshot.particles.body)+1
    o_dict = {body: orient for body, orient in zip(
         snapshot.particles.body[:nmol],
         orientation[:nmol]
    )}
    orientation = np.array([o_dict[body] for body in snapshot.particles.body])
    return orientation


def colour_orientation(orientations):
    """Get a colour from an orientation."""
    orientations = orientations % 2*np.pi
    return HEX_VALUES[np.floor(orientations/np.pi*180).astype(int)]
