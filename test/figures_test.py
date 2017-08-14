#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test function from the generation of figures."""

import math

from hypothesis import given
from hypothesis.strategies import floats

from statdyn.figures import colour


@given(floats(min_value=-math.tau, max_value=math.tau))
def test_colour_orientation(orientation):
    """Ensure hex values being returned by colour_orientation."""
    int(colour.colour_orientation(orientation)[1:], 16)
