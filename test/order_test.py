#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Ensure correctness of the order parameters."""

import gsd.hoomd
import numpy as np
import pytest
from sdanalysis import order  # type: ignore

INFILES = [
    'test/data/Trimer-13.50-0.40-p2.gsd',
    'test/data/Trimer-13.50-0.40-p2gg.gsd',
    'test/data/Trimer-13.50-0.40-pg.gsd'
]

ML_MODELS = [order.dt_model, order.knn_model]

@pytest.fixture(scope='module', params=INFILES)
def frame(request):
    with gsd.hoomd.open(request.param, 'rb') as f:
        yield f[0]


def test_neighbour_tree(frame):
    order.compute_neighbour_tree(
        frame.configuration.box,
        frame.particles.position
    )
    assert True


def test_compute_neighbours(frame):
    max_radius = 10
    max_neighbours = 6
    num_mols = frame.particles.N
    neighs = order.compute_neighbours(
        frame.configuration.box,
        frame.particles.position,
        max_radius,
        max_neighbours
    )
    assert neighs.shape == (num_mols, max_neighbours)
    assert np.all(neighs < num_mols)


def test_num_neighbours(frame):
    max_radius = 3.5
    neighs = order.num_neighbours(
        frame.configuration.box,
        frame.particles.position,
        max_radius
    )
    assert np.all(neighs == 6)


def test_voronoi_neighbours(frame):
    neighs = order.compute_voronoi_neighs(
        frame.configuration.box,
        frame.particles.position,
    )
    assert np.all(neighs == 6)


def test_orientational_order(frame):
    max_radius = 3.5
    orient_order = order.orientational_order(
        frame.configuration.box,
        frame.particles.position,
        frame.particles.orientation,
        max_radius
    )
    assert np.all(orient_order > 0.60)


def test_relative_orientations(frame):
    orientations = order.relative_orientations(
        frame.configuration.box,
        frame.particles.position,
        frame.particles.orientation,
    )
    assert np.all(np.isfinite(orientations))


@pytest.mark.parametrize('model', ML_MODELS)
def test_ml_models(frame, model):
    ordering = order.compute_ml_order(
        model(),
        frame.configuration.box,
        frame.particles.position,
        frame.particles.orientation,
    )
    assert np.all(ordering != 0)
