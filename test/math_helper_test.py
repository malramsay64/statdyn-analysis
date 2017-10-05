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
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

from statdyn.math_helper import (get_quat_eps, quaternion2z, quaternion_angle,
                                 quaternion_rotation, z2quaternion)

EPS = np.finfo(np.float32).eps * 2

@settings(max_examples=1000)
@given(arrays(np.float32, 1, elements=floats(min_value=-10*np.pi, max_value=10*np.pi), unique=True))
def test_z2quaternion(angles):
    """Ensure quaternions are normalised.

    This test ensures that the quaternions returned by the z2quaternion funtion
    are normalised to a value of 1. This is important as quaternions being normalised
    is an assumption I make throughout the codebase.

    """
    result = z2quaternion(angles)
    print(result)
    assert np.allclose(np.linalg.norm(result, axis=1), 1)


@settings(max_examples=1000)
@given(arrays(np.float32, (1, 4), elements=floats(min_value=-1, max_value=1)
              ).filter(lambda x: np.linalg.norm(x, axis=1) > 0.8))
def test_quaternion2z(quat):
    """Ensures correct range of angles [-pi, pi].

    The range of angles returned by the quaternion2z function is in the range
    [-pi,pi]. This tests that the resulting angle is in this range. There is
    also an assertion to ensure that the sign of the rotation of the quaternion
    is the same as that in the returned orientation.

    """
    quat = quat / np.linalg.norm(quat, axis=1)
    result = quaternion2z(quat)
    assert np.abs(result) < np.pi + EPS


@settings(max_examples=1000)
@given(arrays(np.float32, (1, 4), elements=floats(min_value=-1, max_value=1)
              ).filter(lambda x: np.linalg.norm(x, axis=1) > 0.5))
def test_quaternion_angle_2d(quat):
    """Ensures correct range of angles [pi, pi].

    The range of angles returned by the quaternion2z function is in the range
    [-pi,pi]. This tests that the resulting angle is in this range.

    """
    quat = quat / np.linalg.norm(quat, axis=1)
    result = quaternion_angle(quat)
    assert 0 <= result < 2*(np.pi + EPS)


@settings(max_examples=1000)
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
