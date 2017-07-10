#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Testing the crystal class of statdyn."""

import hoomd
import numpy as np
import pytest
from statdyn import crystals
from statdyn.simulation import initialise

TEST_CLASSES = [
    crystals.Crystal,
    crystals.CrysTrimer,
    crystals.TrimerP2
]

CELL_DIMS = [
    (30, 40),
]


@pytest.mark.parametrize("crys_class", TEST_CLASSES)
def test_init(crys_class):
    """Check the class will initialise."""
    crys_class()


@pytest.mark.parametrize("crys_class", TEST_CLASSES)
def test_get_orientations(crys_class):
    """Test the orientation is returned as a float."""
    crys = crys_class()
    orient = crys.get_orientations()
    assert orient.dtype == float


@pytest.mark.parametrize("crys_class", TEST_CLASSES)
def test_get_unitcell(crys_class):
    """Test the return type is correct."""
    crys = crys_class()
    assert isinstance(crys.get_unitcell(), hoomd.lattice.unitcell)


@pytest.mark.parametrize("crys_class", TEST_CLASSES)
def test_compute_volume(crys_class):
    """Test the return type of the volume computation."""
    crys = crys_class()
    assert isinstance(crys.compute_volume(), float)


@pytest.mark.parametrize("crys_class", TEST_CLASSES)
def test_abs_positions(crys_class):
    """Check the absolute positions function return corectly shaped matrix."""
    crys = crys_class()
    assert crys.get_abs_positions().shape == np.array(crys.positions).shape


def get_distance(pos_a, pos_b, box):
    """Compute the periodic distance between two numpy arrays."""
    ortho_box = np.array((box.Lx, box.Ly, box.Lz))
    delta_x = pos_b - pos_a
    delta_x -= ortho_box * (delta_x > ortho_box * 0.5)
    delta_x += ortho_box * (delta_x <= -ortho_box * 0.5)
    return np.sqrt(np.square(delta_x).sum(axis=1))


@pytest.mark.parametrize("cell_dimensions", CELL_DIMS)
def test_cell_dimensions(cell_dimensions):
    """Test cell paramters work properly."""
    snap = initialise.init_from_crystal(crystals.TrimerP2(),
                                        cell_dimensions=cell_dimensions
                                        ).take_snapshot()
    for i in snap.particles.position:
        distances = get_distance(i, snap.particles.position, snap.box) < 1.1
        assert np.sum(distances) <= 3
