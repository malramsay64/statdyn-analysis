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
import quaternion
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.extra.numpy import arrays, floating_dtypes
from hypothesis.strategies import floats
from sdanalysis.math_helper import (get_quat_eps, quaternion2z,
                                    quaternion_angle, quaternion_rotation,
                                    single_quat_rotation, z2quaternion)

np.seterr(invalid='ignore', under='ignore', over='ignore')
EPS = np.finfo(np.float32).eps * 2


def _normalize_quat(q):
    if len(q) == 1:
        q = [0, 0, q]
    mq = quaternion.from_rotation_vector(q)
    return quaternion.as_float_array(mq.normalized()).astype(np.float32)


def _complex_to_quat(c):
    q = np.zeros(4, dtype=np.float32)
    q[0] = c.real
    q[3] = c.imag
    return q


def _increase_dims(a):
    return np.expand_dims(a, axis=0)


def unit_quaternion_Z():
    return arrays(floating_dtypes(sizes=32), 1
                  ).map(_normalize_quat).filter(lambda x: not np.any(np.isnan(x))).map(_increase_dims)


def unit_quaternion(num_elements=1):
    return arrays(floating_dtypes(sizes=32), 3,
               ).map(_normalize_quat).filter(lambda x: not np.any(np.isnan(x))).map(_increase_dims)


def angle(num_elements=1):
    theta = arrays(np.float32,
                   num_elements,
                   elements=floats(max_value=np.finfo(np.float32).max,
                                   min_value=np.finfo(np.float32).min,
                                   allow_nan=False,
                                   allow_infinity=False),
                   )
    return theta


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
    q_quat = quaternion.as_quat_array(quat)[0]
    q_res = np.array(quaternion.as_rotation_vector(q_quat)[2], dtype=np.float32)
    if q_res > np.pi:
        q_res -= 2*np.pi
    if q_res < -np.pi:
        q_res += 2*np.pi
    print(q_res, result)
    assert (np.isclose(q_res, result, atol=0.1) or
            np.isclose(q_res, result - 2*np.pi, atol=0.1) or
            np.isclose(q_res, result + 2*np.pi, atol=0.1))


@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(unit_quaternion())
def test_quaternion_angle_2d(quat):
    """Ensures correct range of angles [pi, pi].

    The range of angles returned by the quaternion2z function is in the range
    [-pi,pi]. This tests that the resulting angle is in this range.

    """
    assume(not np.isnan(np.linalg.norm(quat, axis=1)))
    result = quaternion_angle(quat)
    assert 0 <= result < 2*(np.pi + EPS)


@given(arrays(np.float32, 1, elements=floats(min_value=-np.pi, max_value=np.pi)))
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
    # print('Result: ', np.cos(result), 'diff: ', angles-result)
    assert np.allclose(np.cos(result), np.cos(angles), atol=2*get_quat_eps())
    assert np.allclose(np.cos(result_angle), np.cos(angles), atol=2*get_quat_eps())


@pytest.mark.parametrize('quaternion, angle', [
    ([ 0.98246199,  0.        ,  0.        ,  0.18646298],  0.37512138),
    ([ 0.20939827,  0.        ,  0.        ,  0.97783041],  2.7196734 ),
    ([ 0.97660005,  0.        ,  0.        , -0.21506353], -0.43351361),
    ([-0.21179545,  0.        ,  0.        ,  0.977314  ], -2.71476936),
])
def test_quaternion2z_specifics(quaternion, angle):
    """Test a few specific examples of converting a quaternion to an angle.

    The four values here are the angles in the p2gg crystal structure of the trimer,
    which are also in the four different quadrants of the 2d plane.
    """
    assert np.allclose(quaternion2z(np.array([quaternion], dtype=np.float32)), angle, atol=get_quat_eps())


def test_quat_rotation():
    initial = np.random.random((1000, 4)).astype(np.float32)
    final = np.random.random((1000, 4)).astype(np.float32)
    initial = initial / np.linalg.norm(initial, axis=1).reshape(1000, 1)
    final = final / np.linalg.norm(final, axis=1).reshape(1000, 1)
    result = np.empty(1000, dtype=np.float32)
    quaternion_rotation(initial, final, result)
    assert np.all(result < 2*np.pi)
    assert np.all(0 < result)
    initial_q = quaternion.as_quat_array(initial)
    final_q = quaternion.as_quat_array(final)
    result_q = np.array([quaternion.rotation_intrinsic_distance(i, f)
                           for i, f in zip(initial_q, final_q)], dtype=np.float32)
    assert np.allclose(result, result_q, atol=2e-5)


def test_quaternion_zero_rotation():
    initial = np.array([[1, 0, 0, 0]], dtype=np.float32)
    result = np.empty(1, dtype=np.float32)
    quaternion_rotation(initial, initial, result)
    assert result == 0


def test_quaternion_small_rotation():
    """Small rotations should show up as no rotation."""
    initial = np.array([[1, 0, 0, 0]], dtype=np.float32)
    final = z2quaternion(np.array([1e-3], dtype=np.float32))
    result = np.empty(1, dtype=np.float32)
    quaternion_rotation(initial, final, result)
    assert result == 0


def test_single_quaternion_zero_rotation():
    initial = np.array([[1, 0, 0, 0]], dtype=np.float32)
    assert single_quat_rotation(initial, 0, 0) == 0


def test_single_quaternion_small_rotation():
    """Small rotations should show up as no rotation."""
    initial = np.append(np.array([[1, 0, 0, 0]], dtype=np.float32),
                        z2quaternion(np.array([1e-3], dtype=np.float32)),
                        axis=0)
    assert single_quat_rotation(initial, 0, 1) == 0
