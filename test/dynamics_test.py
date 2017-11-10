#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Testing the dynamics module."""

import numpy as np
import quaternion
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

from statdyn.analysis import dynamics
from statdyn.analysis.read import process_gsd

from .math_helper_test import unit_quaternion

MAX_BOX = 20.

DTYPE = np.float32
EPS = 2*np.sqrt(np.finfo(DTYPE).eps)
HYP_DTYPE = DTYPE


def translationalDisplacement_reference(box: np.ndarray,
                                        initial: np.ndarray,
                                        final: np.ndarray,
                                        ) -> np.ndarray:
    """Simplified reference implementation for computing the displacement.

    This computes the displacment using the shortest path from the original
    position to the final position.

    """
    result = np.empty(final.shape[0], dtype=final.dtype)
    for index in range(len(result)):
        temp = initial[index] - final[index]
        for i in range(3):
            if temp[i] > box[i]/2:
                temp[i] -= box[i]
            if temp[i] < -box[i]/2:
                temp[i] += box[i]
        result[index] = np.linalg.norm(temp, axis=0)
    return result


def rotationalDisplacement_reference(initial: np.ndarray,
                                     final: np.ndarray,
                                     ) -> np.ndarray:
    """Simplified reference implementation of the rotational displacement."""
    init = quaternion.as_quat_array(initial)[0]
    fin = quaternion.as_quat_array(final)[0]
    return quaternion.rotation_intrinsic_distance(init, fin)


@given(arrays(HYP_DTYPE, (10, 3), elements=floats(-MAX_BOX/4, MAX_BOX/4)),
       arrays(HYP_DTYPE, (10, 3), elements=floats(-MAX_BOX/4, MAX_BOX/4)))
def test_translationalDisplacement_noperiod(init, final):
    """Test calculation of the translational displacement.

    This test ensures that the result is close to the numpy.linalg.norm
    function in the case where there is no periodic boundaries to worry
    about.
    """
    box = np.array([MAX_BOX, MAX_BOX, MAX_BOX], dtype=DTYPE)
    np_res = np.linalg.norm(init-final, axis=1)
    result = dynamics.translationalDisplacement(box, init, final)
    ref_res = translationalDisplacement_reference(box, init, final)
    print(result)
    assert np.allclose(result, np_res, atol=EPS)
    assert np.allclose(result, ref_res, atol=EPS)


@given(arrays(HYP_DTYPE, (10, 3), elements=floats(-MAX_BOX/2, -MAX_BOX/4-1e-5)),
       arrays(HYP_DTYPE, (10, 3), elements=floats(MAX_BOX/4, MAX_BOX/2)))
def test_translationalDisplacement_periodicity(init, final):
    """Ensure the periodicity is calulated appropriately.

    This is testing that periodic boundaries are identified appropriately.
    """
    box = np.array([MAX_BOX, MAX_BOX, MAX_BOX], dtype=DTYPE)
    np_res = np.square(np.linalg.norm(init-final, axis=1))
    result = dynamics.translationalDisplacement(box, init, final)
    ref_res = translationalDisplacement_reference(box, init, final)
    assert np.all(np.logical_not(np.isclose(result, np_res)))
    assert np.allclose(result, ref_res, atol=EPS)


@given(arrays(HYP_DTYPE, (10, 3), elements=floats(-MAX_BOX/2, MAX_BOX/2)),
       arrays(HYP_DTYPE, (10, 3), elements=floats(-MAX_BOX/2, MAX_BOX/2)))
def test_translationalDisplacement(init, final):
    """Ensure the periodicity is calulated appropriately.

    This is testing that periodic boundaries are identified appropriately.
    """
    box = np.array([MAX_BOX, MAX_BOX, MAX_BOX], dtype=DTYPE)
    result = dynamics.translationalDisplacement(box, init, final)
    ref_res = translationalDisplacement_reference(box, init, final)
    assert np.allclose(result, ref_res, atol=EPS)


@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(unit_quaternion(), unit_quaternion())
def test_rotationalDisplacement(init, final):
    """Test the calculation of the rotationalDisplacement.

    This compares the result of my algorithm to the quaternion library
    which is much slower on arrays of values.
    """
    assume(not np.any(np.isnan(init)))
    assume(not np.any(np.isnan(final)))
    result = dynamics.rotationalDisplacement(init, final)
    ref_res = rotationalDisplacement_reference(init, final)
    assert np.allclose(result, ref_res, equal_nan=True, atol=10e-2)


@given(arrays(HYP_DTYPE, (100), elements=floats(0, 10)))
def test_alpha(displacement):
    """Test the computation of the non-gaussian parameter."""
    alpha = dynamics.alpha_non_gaussian(displacement)
    assume(not np.isnan(alpha))
    assert alpha >= -1


@given(arrays(HYP_DTYPE, (100), elements=floats(0, 10)),
       arrays(HYP_DTYPE, (100), elements=floats(0, 2*np.pi)))
def test_overlap(displacement, rotation):
    """Test the computation of the overlap of the largest values."""
    overlap_same = dynamics.mobile_overlap(rotation, rotation)
    assert np.isclose(overlap_same, 1)
    overlap = dynamics.mobile_overlap(displacement, rotation)
    assert 0. <= overlap <= 1.


@given(arrays(HYP_DTYPE, (100), elements=floats(0, 10)),
       arrays(HYP_DTYPE, (100), elements=floats(0, 2*np.pi)))
def test_spearman_rank(displacement, rotation):
    """Test the spearman ranking coefficient."""
    spearman_same = dynamics.spearman_rank(rotation, rotation)
    assert np.isclose(spearman_same, 1)
    spearman = dynamics.spearman_rank(rotation, rotation)
    assert -1 <= spearman <= 1

def test_dynamics():
    process_gsd('test/data/trajectory-13.50-3.00.gsd')

def test_molecularRelaxation():
    num_elements = 10
    tau_1 = dynamics.molecularRelaxation(num_elements, 2.)
    tau_1.add(120, np.zeros(num_elements))
    assert np.all(tau_1.get_status() == np.full(num_elements, tau_1._max_value))
    tau_1.add(100, np.ones(num_elements) * 3)
    assert np.all(tau_1.get_status() == np.full(num_elements, 100))
