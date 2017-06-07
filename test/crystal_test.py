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

from statdyn import crystals
import hoomd
import numpy as np
import quaternion as qt
import pytest

test_classes = [
    crystals.Crystal,
    crystals.CrysTrimer,
    crystals.p2
]


@pytest.mark.parametrize("crys_class", test_classes)
def test_init(crys_class):
    crys_class()

@pytest.mark.parametrize("crys_class", test_classes)
def test_get_orientations(crys_class):
    crys = crys_class()
    orient = crys.get_orientations()
    assert orient.dtype == float

@pytest.mark.parametrize("crys_class", test_classes)
def test_get_unitcell(crys_class):
    crys = crys_class()
    assert isinstance(crys.get_unitcell(), hoomd.lattice.unitcell)

@pytest.mark.parametrize("crys_class", test_classes)
def test_compute_volume(crys_class):
    crys = crys_class()
    assert isinstance(crys.compute_volume(), float)

@pytest.mark.parametrize("crys_class", test_classes)
def test_abs_positions(crys_class):
    crys = crys_class()
    assert crys.get_abs_positions().shape == np.array(crys.positions).shape
