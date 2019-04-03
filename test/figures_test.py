#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
#
# pylint: disable=redefined-outer-name
#

"""Test function from the generation of figures."""

import math

import bokeh.colors
import gsd.hoomd
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats

from sdanalysis.figures.configuration import colour_orientation, plot_frame
from sdanalysis.frame import HoomdFrame
from sdanalysis.order import compute_voronoi_neighs


@given(floats(min_value=-math.pi, max_value=math.pi))
def test_colour_orientation(orientation):
    """Ensure Color objects values being returned by colour_orientation."""
    c = colour_orientation(orientation)
    assert isinstance(c, bokeh.colors.Color)


@pytest.fixture()
def snapshot():
    with gsd.hoomd.open("test/data/trajectory-Trimer-P13.50-T3.00.gsd") as trj:
        yield HoomdFrame(trj[0])


def test_plot_frame(snapshot, mol):
    plot_frame(snapshot, molecule=mol)


def test_plot_frame_orderlist(snapshot, mol):
    order_list = np.random.choice([0, 1, 2], len(snapshot))
    plot_frame(snapshot, molecule=mol, order_list=order_list)


def test_plot_frame_orderfunc(snapshot, mol):
    def compute_neigh_ordering(box, positions, _):
        return compute_voronoi_neighs(box, positions) == 6

    plot_frame(snapshot, molecule=mol, order_function=compute_neigh_ordering)


@pytest.mark.parametrize("dtype", [int, str, float])
def test_plot_frame_categorical(snapshot, mol, dtype):
    categories = np.random.choice([0, 1, 2], len(snapshot)).astype(dtype)
    plot_frame(snapshot, molecule=mol, order_list=categories, categorical_colour=True)


def test_order(snapshot):
    order_list = compute_voronoi_neighs(snapshot.box, snapshot.position)
    plot_frame(snapshot, order_list=order_list)
