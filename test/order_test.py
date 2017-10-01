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
from hypothesis import given
from hypothesis.strategies import floats

# Allowing for more interactive updating of tests running cython code
try:
    import pyximport
    pyximport.install()
    from statdyn.analysis import order
except ModuleNoeFoundError:
    from statdyn.analysis import order


def z_angle_to_quat(theta):
    return np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)])


@given(floats(min_value=-2*np.pi, max_value=2*np.pi))
def test_z_orientation(angle):
    """Roundtrip the z orientation.

    This is a basic test to ensure that the z orientation is being calculated
    appropriately, primarily that it gives the appropriate value.
    """
    angle_array = np.array([z_angle_to_quat(angle)], dtype=np.float32)
    computed_angle = order.get_z_orientation(angle_array)
    if angle <= -np.pi:
        angle += 2*np.pi
    elif angle > np.pi:
        angle -= 2*np.pi
    assert np.isclose(computed_angle[0], angle, atol=1e-6)

@pytest.mark.parametrize('infile', [
    'test/data/Trimer-13.50-0.40-p2.gsd',
    'test/data/Trimer-13.50-0.40-p2gg.gsd',
    'test/data/Trimer-13.50-0.40-pg.gsd'
])
def test_nearest_neighbours(infile):
    with gsd.hoomd.open(infile, 'rb') as f:
        frame = f[0]
        max_radius = 10
        max_neighbours = 6
        num_mols = frame.particles.N
        box = frame.configuration.box
        simulation_box = Box(box[0], box[1], is2D=True)
        nn = NearestNeighbors(rmax=max_radius, n_neigh=max_neighbours)
        nn.compute(simulation_box, frame.particles.position, frame.particles.position)
    for i in range(num_mols):
        assert np.all(nn.getNeighbors(i) < num_mols)


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
        neighs = compute_neighbours(frame.configuration.box,
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
        neighs = order.num_neighbours(frame.configuration.box,
                                      frame.particles.position,
                                      frame.particles.orientation,
                                      max_radius)
    assert np.all(neighs == 6)


def test_orientational_order():
    with gsd.hoomd.open('test/data/Trimer-13.50-0.40-p2.gsd') as f:
        frame = f[0]
        max_radius = 3.5
        orient_order = order.orientational_order(frame.configuration.box,
                                                 frame.particles.position,
                                                 frame.particles.orientation,
                                                 max_radius)
    assert np.allclose(orient_order, 1, atol=0.02)
