#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the molecule class."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats

from sdanalysis import molecules


def test_scale_moment_inertia(mol):
    scale_factor = 10.
    mol.moment_inertia_scale = scale_factor
    assert mol.moment_inertia_scale == scale_factor


def test_get_radii(mol):
    radii = mol.get_radii()
    assert radii[0] == 1.


def test_get_types(mol):
    types = mol.get_types()
    assert "R" not in types
