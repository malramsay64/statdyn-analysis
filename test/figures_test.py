#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
#
#

"""Test function from the generation of figures."""

import math

import bokeh.colors
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats

from sdanalysis.figures.configuration import colour_orientation, plot_frame
from sdanalysis.order import create_neigh_ordering


@given(floats(min_value=-math.pi, max_value=math.pi))
def test_colour_orientation(orientation):
    """Ensure Color objects values being returned by colour_orientation."""
    c = colour_orientation(orientation)
    assert isinstance(c, bokeh.colors.Color)


def test_plot_frame(frame, mol):
    plot_frame(frame, molecule=mol)


@pytest.mark.parametrize("categories", [True, False])
def test_plot_frame_orderlist(frame, mol, categories):
    order_list = np.random.choice([0, 1, 2], len(frame))
    plot_frame(
        frame, molecule=mol, order_list=order_list, categorical_colour=categories
    )


@pytest.mark.parametrize("categories", [True, False])
def test_plot_frame_orderfunc(frame, mol, categories):
    plot_frame(
        frame,
        molecule=mol,
        order_function=create_neigh_ordering(6),
        categorical_colour=categories,
    )


@pytest.mark.parametrize("dtype", [int, str, float])
def test_plot_frame_categorical(frame, mol, dtype):
    categories = np.random.choice([0, 1, 2], len(frame)).astype(dtype)
    plot_frame(frame, molecule=mol, order_list=categories, categorical_colour=True)


def test_order(frame):
    order_func = create_neigh_ordering(6)
    order_list = order_func(frame)
    plot_frame(frame, order_list=order_list)
