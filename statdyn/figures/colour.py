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
from hsluv import hpluv_to_hex

from ..analysis.order import get_z_orientation

logger = logging.getLogger(__name__)

HEX_VALUES_DARK = np.array([hpluv_to_hex((value, 85, 65)) for value in range(360)])
HEX_VALUES_LIGHT = np.array([hpluv_to_hex((value, 85, 85)) for value in range(360)])


def clean_orientation(snapshot):
    """Convert an orientation to a sensible format."""
    orientations = get_z_orientation(snapshot.particles.orientation)
    nmol = max(snapshot.particles.body)+1
    o_dict = {body: orient for body, orient in zip(
        snapshot.particles.body[:nmol],
        orientations
    )}
    orientation = np.array([o_dict[body] for body in snapshot.particles.body])
    return orientation


def colour_orientation(orientations, light_colours=False):
    """Get a colour from an orientation."""
    orientations = orientations % 2 * np.pi
    if light_colours:
        return HEX_VALUES_LIGHT[np.floor(orientations / np.pi * 180).astype(int)]
    return HEX_VALUES_DARK[np.floor(orientations / np.pi * 180).astype(int)]
