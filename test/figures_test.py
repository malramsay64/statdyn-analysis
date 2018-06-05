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
import pytest
from hypothesis import given
from hypothesis.strategies import floats

from sdanalysis.figures.configuration import colour_orientation, plot_frame
from sdanalysis.frame import gsdFrame
from sdanalysis.molecules import Trimer
from sdanalysis.order import compute_voronoi_neighs


@given(floats(min_value=-math.pi, max_value=math.pi))
def test_colour_orientation(orientation):
    """Ensure Color objects values being returned by colour_orientation."""
    colour_orientation(orientation)


@pytest.mark.parametrize("molecule", [None, Trimer()])
def test_plot_frame(molecule):
    with gsd.hoomd.open("test/data/trajectory-Trimer-P13.50-T3.00.gsd") as trj:
        plot_frame(gsdFrame(trj[0]), molecule=molecule)


def test_order():
    with gsd.hoomd.open("test/data/trajectory-Trimer-P13.50-T3.00.gsd") as trj:
        order_list = compute_voronoi_neighs(
            trj[0].configuration.box, trj[0].particles.position
        )
        plot_frame(gsdFrame(trj[0]), order_list=order_list)
