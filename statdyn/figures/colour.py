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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.WARNING)

HEX_VALUES = np.array([hpluv_to_hex((value, 85, 65)) for value in range(360)])


def colour_orientation(orientations):
    """Get a colour from an orientation."""
    orientations = orientations % 2*np.pi
    return HEX_VALUES[np.floor(orientations/np.pi*180).astype(int)]
