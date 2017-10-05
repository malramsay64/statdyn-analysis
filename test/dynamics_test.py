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

from statdyn.analysis import dynamics

MAX_BOX = 20.


def translationalDisplacement_reference(box: np.ndarray,
                                        initial: np.ndarray,
                                        final: np.ndarray,
                                        result: np.ndarray
                                        ) -> None:
    """Simplified reference implementation for computing the displacement.

    This computes the displacment using the shortest path from the original
    position to the final position.

    """
    for index in range(len(result)):
        temp = initial[index] - final[index]
        for i in range(3):
            if temp[i] > box[i]/2:
                temp[i] -= box[i]
            if temp[i] < -box[i]/2:
                temp[i] += box[i]
        result[index] = np.linalg.norm(temp)


def rotationalDisplacement_reference(initial: np.ndarray,
                                     final: np.ndarray,
                                     result: np.ndarray,
                                     ) -> None:
    """Simplified reference implementation of the rotational displacement."""
    for index in range(len(result)):
        result[index] = 2*np.arccos(np.abs(np.dot(initial[index], final[index])))


@given(arrays(np.float64, (10, 3), elements=floats(-MAX_BOX/4, MAX_BOX/4)),
       arrays(np.float64, (10, 3), elements=floats(-MAX_BOX/4, MAX_BOX/4)))
def test_translationalDisplacement_noperiod(init, final):
    """Test calculation of the translational displacement.

    This test ensures that the result is close to the numpy.linalg.norm
    function in the case where there is no periodic boundaries to worry
    about.
    """
    box = np.array([MAX_BOX, MAX_BOX, MAX_BOX])
    result = np.zeros(len(init))
    ref_res = np.zeros(len(init))
    np_res = np.linalg.norm(init-final, axis=1)
    dynamics.translationalDisplacement(box, init, final, result)
    translationalDisplacement_reference(box, init, final, ref_res)
    assert np.allclose(result, np_res)
    assert np.allclose(result, ref_res)


@given(arrays(np.float64, (10, 3), elements=floats(-MAX_BOX/2, -MAX_BOX/4-1e-5)),
       arrays(np.float64, (10, 3), elements=floats(MAX_BOX/4, MAX_BOX/2)))
def test_translationalDisplacement_periodicity(init, final):
    """Ensure the periodicity is calulated appropriately.

    This is testing that periodic boundaries are identified appropriately.
    """
    box = np.array([MAX_BOX, MAX_BOX, MAX_BOX])
    result = np.empty(len(init))
    ref_res = np.empty(len(init))
    np_res = np.square(np.linalg.norm(init-final, axis=1))
    dynamics.translationalDisplacement(box, init, final, result)
    translationalDisplacement_reference(box, init, final, ref_res)
    assert np.all(np.logical_not(np.isclose(result, np_res)))
    assert np.allclose(result, ref_res)


@given(arrays(np.float64, (10, 3), elements=floats(-MAX_BOX/2, MAX_BOX/2)),
       arrays(np.float64, (10, 3), elements=floats(-MAX_BOX/2, MAX_BOX/2)))
def test_translationalDisplacement(init, final):
    """Ensure the periodicity is calulated appropriately.

    This is testing that periodic boundaries are identified appropriately.
    """
    box = np.array([MAX_BOX, MAX_BOX, MAX_BOX])
    result = np.empty(len(init))
    ref_res = np.empty(len(init))
    dynamics.translationalDisplacement(box, init, final, result)
    translationalDisplacement_reference(box, init, final, ref_res)
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
    result = np.zeros(init.shape[0])
    ref_res = np.empty(init.shape[0])
    dynamics.rotationalDisplacement(init, final, result)
    rotationalDisplacement_reference(init, final, ref_res)
    assert np.allclose(result, ref_res, equal_nan=True)


@given(arrays(np.float64, (100), elements=floats(0, 10)))
def test_alpha(displacement):
    """Test the computation of the non-gaussian parameter."""
    alpha = dynamics.alpha_non_gaussian(displacement)
    assert alpha >= -1


@given(arrays(np.float64, (100), elements=floats(0, 10)),
       arrays(np.float64, (100), elements=floats(0, 2*np.pi)))
def test_overlap(displacement, rotation):
    """Test the computation of the overlap of the largest values."""
    overlap_same = dynamics.mobile_overlap(rotation, rotation)
    assert np.isclose(overlap_same, 1)
    overlap = dynamics.mobile_overlap(displacement, rotation)
    assert 0. <= overlap <= 1.


@given(arrays(np.float64, (100), elements=floats(0, 10)),
       arrays(np.float64, (100), elements=floats(0, 2*np.pi)))
def test_spearman_rank(displacement, rotation):
    """Test the spearman ranking coefficient."""
    spearman_same = dynamics.spearman_rank(rotation, rotation)
    assert np.isclose(spearman_same, 1)
    spearman = dynamics.spearman_rank(rotation, rotation)
    assert -1 <= spearman <= 1
