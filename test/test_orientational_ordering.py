#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the orientational ordering function

This is a function to calculate the orientational ordering of a local configuration
based on the orientation of the nearest neighbours.

"""

import hypothesis.strategies as st
import numpy as np
import rowan
from hypothesis import assume, given

from sdanalysis.order import _orientational_order


def test_parallel():
    neighbourlist = np.array([[1], [0]])
    orientation = np.zeros((2, 4))
    orientation[:, 0] = 1

    result = _orientational_order(neighbourlist, orientation)

    assert np.allclose(result, 1)


def test_antiparallel():
    neighbourlist = np.array([[1], [0]])
    orientation = np.zeros((2, 4))
    orientation[:, 0] = 1
    orientation[0, :] = [0, 0, 0, 1]

    result = _orientational_order(neighbourlist, orientation)

    assert np.allclose(result, 1)


def test_perpendicular():
    neighbourlist = np.array([[1], [0]])
    orientation = np.zeros((2, 4))
    orientation[:, 0] = 1
    orientation[0, :] = rowan.from_euler(np.pi / 2, 0, 0)

    result = _orientational_order(neighbourlist, orientation)

    assert np.allclose(result, 0)


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_properties(angle):
    assume(np.cos(angle))
    neighbourlist = np.array([[1], [0]])
    orientation = np.array([rowan.from_euler(0, 0, 0), rowan.from_euler(angle, 0, 0)])

    result = _orientational_order(neighbourlist, orientation)

    assert np.allclose(result, np.square(np.cos(angle)))
