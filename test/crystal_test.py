#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Testing the crystal class of statdyn
"""

import hoomd
import numpy as np
import pytest
from statdyn import crystals

TEST_CLASSES = [
    crystals.Crystal,
    crystals.CrysTrimer,
    crystals.TrimerP2
]


@pytest.mark.parametrize("crys_class", TEST_CLASSES)
def test_init(crys_class):
    """Check the class will initialise"""
    crys_class()


@pytest.mark.parametrize("crys_class", TEST_CLASSES)
def test_get_orientations(crys_class):
    """Test the orientation is returned as a float"""
    crys = crys_class()
    orient = crys.get_orientations()
    assert orient.dtype == float


@pytest.mark.parametrize("crys_class", TEST_CLASSES)
def test_get_unitcell(crys_class):
    """Test that the return type is correct"""
    crys = crys_class()
    assert isinstance(crys.get_unitcell(), hoomd.lattice.unitcell)


@pytest.mark.parametrize("crys_class", TEST_CLASSES)
def test_compute_volume(crys_class):
    """Test the return type of the volume computation"""
    crys = crys_class()
    assert isinstance(crys.compute_volume(), float)


@pytest.mark.parametrize("crys_class", TEST_CLASSES)
def test_abs_positions(crys_class):
    """Check that the absolute positions function returns a matrix of the
    correct shape
    """
    crys = crys_class()
    assert crys.get_abs_positions().shape == np.array(crys.positions).shape
