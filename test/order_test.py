#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
#
# pylint: disable=redefined-outer-name
#

"""Ensure correctness of the order parameters."""

import numpy as np
import pytest
from sdanalysis import order, read

INFILES = [
    "test/data/dump-Trimer-13.50-0.40-p2.gsd",
    "test/data/dump-Trimer-13.50-0.40-p2gg.gsd",
    "test/data/dump-Trimer-13.50-0.40-pg.gsd",
]


@pytest.fixture(scope="module", params=INFILES)
def frame(request):
    return next(read.open_trajectory(request.param))


def test_compute_neighbours(frame):
    max_radius = 10
    max_neighbours = 6
    num_mols = len(frame)
    neighs = order.compute_neighbours(
        frame.box, frame.position, max_radius, max_neighbours
    )
    assert neighs.shape == (num_mols, max_neighbours)
    assert np.all(neighs < num_mols)


def test_num_neighbours(frame):
    max_radius = 3.5
    neighs = order.num_neighbours(frame.box, frame.position, max_radius)
    assert np.all(neighs == 6)


def test_voronoi_neighbours(frame):
    neighs = order.compute_voronoi_neighs(frame.box, frame.position)
    assert np.all(neighs == 6)


def test_create_neigh_ordering(frame):
    order_func = order.create_neigh_ordering(6)
    assert np.all(order_func(frame))


def test_orientational_order(frame):
    max_radius = 3.5
    orient_order = order.orientational_order(
        frame.box, frame.position, frame.orientation, max_radius
    )
    assert np.all(orient_order > 0.60)


def test_create_orient_ordering(frame):
    order_func = order.create_orient_ordering(0.60)
    assert np.all(order_func(frame))


def test_relative_distance(frame):
    print(frame.box)
    distances = order.relative_distances(frame.box, frame.position)
    assert np.all(np.isfinite(distances))


def test_relative_orientations(frame):
    orientations = order.relative_orientations(
        frame.box, frame.position, frame.orientation
    )
    assert np.all(np.isfinite(orientations))
