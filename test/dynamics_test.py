#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Testing the dynamics module."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from quaternion import (as_float_array, as_quat_array,
                        rotation_intrinsic_distance)

from statdyn.analysis import dynamics

MAX_BOX = 20.

def squaredDisplacement_reference(box: np.ndarray,
                                  initial: np.ndarray,
                                  final: np.ndarray,
                                  result: np.ndarray
                                  ) -> None:
    """Simple implementation of function for computing the squared displacement.

    This computes the displacment using the shortest path from the original
    position to the final position. This is a reasonable assumption to make
    since the path
    """
    for index in range(len(result)):
        temp = initial[index] - final[index]
        for i in range(3):
            if temp[i] > box[i]/2:
                temp[i] -= box[i]
            if temp[i] < -box[i]/2:
                temp[i] += box[i]
        result[index] = np.square(temp).sum()

@given(arrays(np.float64, (10, 3), elements=floats(-MAX_BOX/4, MAX_BOX/4)),
       arrays(np.float64, (10, 3), elements=floats(-MAX_BOX/4, MAX_BOX/4)))
def test_sq_displacement(init, final):
    """Test calculation of the squared displacement.

    This test ensures that the result is close to the numpy.linalg.norm
    function in the case where there is no periodic boundaries to worry
    about.
    """
    box = np.array([MAX_BOX, MAX_BOX, MAX_BOX])
    result = np.zeros(len(init))
    ref_res = np.zeros(len(init))
    np_res = np.square(np.linalg.norm(init-final, axis=1))
    dynamics.squaredDisplacement(box, init, final, result)
    squaredDisplacement_reference(box, init, final, ref_res)
    assert np.allclose(result, np_res)
    assert np.allclose(result, ref_res)


@given(arrays(np.float64, (10, 3), elements=floats(-MAX_BOX, -MAX_BOX/2-1e-5)),
       arrays(np.float64, (10, 3), elements=floats(MAX_BOX/2, MAX_BOX)))
def test_sq_displacement_periodicity(init, final):
    """Ensure the periodicity is calulated appropriately.

    This is testing that periodic boundaries are identified appropriately.
    """
    box = np.array([MAX_BOX, MAX_BOX, MAX_BOX])
    result = np.empty(len(init))
    ref_res = np.empty(len(init))
    np_res = np.square(np.linalg.norm(init-final, axis=1))
    dynamics.squaredDisplacement(box, init, final, result)
    squaredDisplacement_reference(box, init, final, ref_res)
    assert np.all(np.logical_not(np.isclose(result, np_res)))
    assert np.allclose(result, ref_res)


@given(arrays(np.float64, (1, 4), elements=floats(-1, 1)
              ).filter(lambda x: np.linalg.norm(x) > 0.5),
       arrays(np.float64, (1, 4), elements=floats(-1, 1)
              ).filter(lambda x: np.linalg.norm(x) > 0.5),)
def test_rotationalDisplacement(init, final):
    """Test the calculation of the rotationalDisplacement.

    This compares the result of my algorithm to the quaternion library
    which is much slower on arrays of values.
    """
    init = init / np.linalg.norm(init, axis=1)
    final = final / np.linalg.norm(final, axis=1)
    init_quat = as_quat_array(init)
    final_quat = as_quat_array(final)
    result = np.zeros(len(init))
    dynamics.rotationalDisplacement(init, final, result)
    quat_res = []
    for i, f in zip(init_quat, final_quat):
        quat_res.append(rotation_intrinsic_distance(i, f))
    assert np.allclose(result, np.array(quat_res), equal_nan=True, atol=5e-6)


@given(arrays(np.float64, (100), elements=floats(0, 10)))
def test_alpha(displacement_squared):
    """Test the computation of the non-gaussian parameter."""
    alpha = dynamics.alpha_non_gaussian(displacement_squared)
    assert alpha >= -1


@given(arrays(np.float64, (100), elements=floats(0, 10)),
       arrays(np.float64, (100), elements=floats(0, 2*np.pi)))
def test_overlap(displacement_squared, rotation):
    """Test the computation of the overlap of the largest values."""
    overlap_same = dynamics.mobile_overlap(rotation, rotation)
    assert np.isclose(overlap_same, 1)
    overlap = dynamics.mobile_overlap(displacement_squared, rotation)
    assert overlap <= 1.
    assert overlap >= 0.


@given(arrays(np.float64, (100), elements=floats(0, 10)),
       arrays(np.float64, (100), elements=floats(0, 2*np.pi)))
def test_spearman_rank(displacement_squared, rotation):
    """Test the spearman ranking coefficient."""
    spearman_same = dynamics.spearman_rank(rotation, rotation)
    assert np.isclose(spearman_same, 1)
    spearman = dynamics.spearman_rank(rotation, rotation)
    assert spearman <= 1
    assert spearman >= -1
