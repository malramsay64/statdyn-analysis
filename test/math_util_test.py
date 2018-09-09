#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the mathematical helper functions."""

import numpy as np
import pytest
import rowan
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.extra.numpy import arrays, floating_dtypes
from hypothesis.strategies import floats
from numpy.testing import assert_allclose

from sdanalysis.math_util import (
    quaternion2z,
    quaternion_angle,
    quaternion_rotation,
    z2quaternion,
)

np.seterr(invalid="ignore", under="ignore", over="ignore")
EPS = np.finfo(np.float32).eps * 2


def _increase_dims(a):
    return np.expand_dims(a, axis=0)


def angle(num_elements=1):
    theta = arrays(
        np.float32,
        num_elements,
        elements=floats(
            max_value=np.finfo(np.float32).max,
            min_value=np.finfo(np.float32).min,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    return theta


def unit_quaternion_Z():
    return angle().map(lambda z: rowan.from_euler(z, 0, 0))


@pytest.fixture
def quat():
    np.random.seed(0)
    return rowan.random.rand(1000)


@given(angle())
def test_z2quaternion(angles):
    """Ensure quaternions are normalised.

    This test ensures that the quaternions returned by the z2quaternion funtion
    are normalised to a value of 1. This is important as quaternions being normalised
    is an assumption I make throughout the codebase.

    """
    result = z2quaternion(angles)
    assert np.allclose(np.linalg.norm(result, axis=1), 1)


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(unit_quaternion_Z())
def test_quaternion2z(quat):
    """Ensures correct range of angles [-pi, pi].

    The range of angles returned by the quaternion2z function is in the range
    [-pi,pi]. This tests that the resulting angle is in this range. There is
    also an assertion to ensure that the sign of the rotation of the quaternion
    is the same as that in the returned orientation.

    Since complex numbers are a subset of the quaternions there is also a test
    to ensure these functions are giving the same angle as for complex numbers.

    """
    result = quaternion2z(quat)
    assert -np.pi <= result
    assert result <= np.pi


def test_quaternion_angle_2d(quat):
    """Ensures correct range of angles [-pi, pi].

    The range of angles returned by the quaternion2z function is in the range
    [-pi,pi]. This tests that the resulting angle is in this range.

    """
    result = quaternion_angle(quat)
    assert np.all(-np.pi <= result)
    assert np.all(result <= np.pi)


@given(arrays(np.float64, 1, elements=floats(min_value=-np.pi, max_value=np.pi)))
def test_angle_roundtrip(angles):
    """Test the roundtrip of conversion from angle to quat and back.

    This roundtrip is a good test of whether I am getting reasonable results from
    the functions. The reason I am taking the cos of both the input and the result
    is that the cos function in the conversion to a quaternion is the limiting factor
    of the precision. Rather than making the tolerance really large I have just
    normalised by the precision of the least precise operation.

    """
    quat = z2quaternion(angles)
    result = quaternion2z(quat)
    result_angle = quaternion_angle(quat)
    assert_allclose(result, angles, atol=1e-7)


@pytest.mark.parametrize(
    "quaternion, angle",
    [
        ([0.98246199, 0., 0., 0.18646298], 0.37512138),
        ([0.20939827, 0., 0., 0.97783041], 2.7196734),
        ([0.97660005, 0., 0., -0.21506353], -0.43351361),
        ([-0.21179545, 0., 0., 0.977314], -2.71476936),
    ],
)
def test_quaternion2z_specifics(quaternion, angle):
    """Test a few specific examples of converting a quaternion to an angle.

    The four values here are the angles in the p2gg crystal structure of the trimer,
    which are also in the four different quadrants of the 2d plane.
    """
    assert np.allclose(quaternion2z(np.array([quaternion], dtype=np.float32)), angle)


def test_quat_rotation():
    np.random.seed(0)
    initial = rowan.random.rand(1000)
    final = rowan.random.rand(1000)
    result = np.empty(1000)
    quaternion_rotation(initial, final, result)
    assert np.all(result < 2 * np.pi)
    assert np.all(0 < result)


def test_quaternion_zero_rotation():
    initial = np.array([[1, 0, 0, 0]], dtype=np.float32)
    result = np.empty(1)
    quaternion_rotation(initial, initial, result)
    assert result == 0


def test_quaternion_small_rotation():
    """Small rotations should show up as no rotation."""
    initial = np.array([[1, 0, 0, 0]])
    final = z2quaternion(np.array([1e-7]))
    result = np.empty(1, dtype=np.float32)
    quaternion_rotation(initial, final, result)
    assert result != 0
