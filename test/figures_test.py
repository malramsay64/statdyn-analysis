#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test function from the generation of figures."""

import math

import gsd.hoomd
from hypothesis import given
from hypothesis.strategies import floats
from sdanalysis.figures import colour
from sdanalysis.figures.configuration import plot, snapshot2data
from sdanalysis.order import compute_voronoi_neighs


@given(floats(min_value=-math.pi, max_value=math.pi))
def test_colour_orientation(orientation):
    """Ensure hex values being returned by colour_orientation."""
    int(colour.colour_orientation(orientation)[1:], 16)

def test_plot():
    with gsd.hoomd.open('test/data/trajectory-13.50-3.00.gsd') as trj:
        plot(trj[0], repeat=True, offset=True)

def test_snapshot2data():
    with gsd.hoomd.open('test/data/trajectory-13.50-3.00.gsd') as trj:
        snapshot2data(trj[0])

def test_order():
    with gsd.hoomd.open('test/data/trajectory-13.50-3.00.gsd') as trj:
        order_list = compute_voronoi_neighs(trj[0].configuration.box,
                                             trj[0].particles.position)
        plot(trj[0], repeat=True, offset=True, order_list=order_list)
