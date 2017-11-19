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

from statdyn.analysis import order  # type: ignore


@pytest.mark.parametrize('infile', [
    'test/data/Trimer-13.50-0.40-p2.gsd',
    'test/data/Trimer-13.50-0.40-p2gg.gsd',
    'test/data/Trimer-13.50-0.40-pg.gsd'
])
def test_compute_neighbours(infile):
    with gsd.hoomd.open(infile, 'rb') as f:
        frame = f[0]
        max_radius = 10
        max_neighbours = 6
        num_mols = frame.particles.N
        neighs = order.compute_neighbours(frame.configuration.box,
                                          frame.particles.position,
                                          max_radius,
                                          max_neighbours)
        assert np.all(neighs < num_mols)


@pytest.mark.parametrize('infile', [
    'test/data/Trimer-13.50-0.40-p2.gsd',
    'test/data/Trimer-13.50-0.40-p2gg.gsd',
    'test/data/Trimer-13.50-0.40-pg.gsd'
])
def test_num_neighbours(infile):
    with gsd.hoomd.open(infile, 'rb') as f:
        frame = f[0]
        max_radius = 3.5
        neighs = order.num_neighbours(
            frame.configuration.box,
            frame.particles.position,
            max_radius
        )
    assert np.all(neighs == 6)


def test_orientational_order():
    with gsd.hoomd.open('test/data/Trimer-13.50-0.40-p2.gsd') as f:
        frame = f[0]
        max_radius = 3.5
        orient_order = order.orientational_order(
            frame.configuration.box,
            frame.particles.position,
            frame.particles.orientation,
            max_radius
        )
    assert np.allclose(orient_order, 1, atol=0.02)
