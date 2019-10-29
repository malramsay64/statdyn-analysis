#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the molecule class."""

import numpy as np
from numpy.testing import assert_allclose

from sdanalysis.molecules import Disc, Sphere


def test_scale_moment_inertia(mol):
    scale_factor = 10.0
    mol.moment_inertia_scale = scale_factor
    assert mol.moment_inertia_scale == scale_factor


def test_get_radii(mol):
    radii = mol.get_radii()
    if isinstance(mol, (Disc, Sphere)):
        assert radii[0] == 0.5
    else:
        assert radii[0] == 1.0


def test_get_types(mol):
    types = mol.get_types()
    assert "R" not in types


def test_position_com(mol):
    assert_allclose(mol.positions.mean(axis=0), np.zeros(3), atol=1e-7)
