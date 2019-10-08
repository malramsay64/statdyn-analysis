#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
#
# pylint: disable=redefined-outer-name
#

"""Test the various helper functions in the package."""

import numpy as np
import pytest
import rowan
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from numpy.testing import assert_allclose
from sdanalysis.params import SimulationParams
from sdanalysis.util import (
    Variables,
    get_filename_vars,
    orientation2positions,
    quaternion2z,
    quaternion_angle,
    quaternion_rotation,
    set_filename_vars,
    z2quaternion,
)

np.seterr(invalid="ignore", under="ignore", over="ignore")
EPS = np.finfo(np.float32).eps * 2


def test_orientation2positions(mol):
    position = np.array([[0, 0, 0]], dtype=np.float32)
    orientation = np.array([[1, 0, 0, 0]], dtype=np.float32)
    x_inv_pos = np.copy(mol.positions)
    x_inv_pos[:, 0] = -x_inv_pos[:, 0]
    rotated_pos = orientation2positions(mol, position, orientation)
    assert np.allclose(rotated_pos, x_inv_pos, atol=1e5)


def test_orientation2positions_invert_xy(mol):
    position = np.array([[0, 0, 0]], dtype=np.float32)
    orientation = np.array([[0, 0, 0, 1]], dtype=np.float32)
    xy_inv_pos = np.copy(mol.positions)
    xy_inv_pos[:, :2] = -xy_inv_pos[:, :2]
    rotated_pos = orientation2positions(mol, position, orientation)
    assert np.allclose(rotated_pos, xy_inv_pos, atol=1e5)


def test_orientation2positions_moved(mol):
    position = np.array([[1, 1, 0]], dtype=np.float32)
    orientation = np.array([[1, 0, 0, 0]], dtype=np.float32)
    rotated_pos = orientation2positions(mol, position, orientation)
    moved_pos = mol.positions + np.repeat(position, mol.num_particles, axis=0)
    assert np.allclose(rotated_pos, moved_pos)


def test_orientation2positions_moved_rot(mol):
    position = np.array([[4, 2, 0]], dtype=np.float32)
    orientation = np.array([[0, 0, 0, 1]], dtype=np.float32)
    rotated_pos = orientation2positions(mol, position, orientation)
    xy_inv_pos = np.copy(mol.positions)
    xy_inv_pos[:, :2] = -xy_inv_pos[:, :2]
    moved_pos = xy_inv_pos + np.tile(position, (mol.num_particles, 1))
    assert np.allclose(rotated_pos, moved_pos)


def test_orientation2positions_moved_rot_multiple(mol):
    position = np.array([[4, 2, 0], [0, 0, 0]], dtype=np.float32)
    orientation = np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32)
    rotated_pos = orientation2positions(mol, position, orientation)
    xy_inv_pos = np.copy(mol.positions)
    xy_inv_pos[:, :2] = -xy_inv_pos[:, :2]
    moved_pos = np.repeat(xy_inv_pos, position.shape[0], axis=0) + np.tile(
        position, (mol.num_particles, 1)
    )
    assert np.allclose(rotated_pos, moved_pos)


@pytest.mark.parametrize("press", ["0.00", "0.50", "1.00", "13.50"])
@pytest.mark.parametrize("temp", ["0.00", "0.10", "1.50", "2.00"])
@pytest.mark.parametrize("crys", [None, "p2", "p2gg", "pg"])
@pytest.mark.parametrize("mol", ["Trimer"])
@pytest.mark.parametrize("prefix", ["dump-", "trajectory-", "thermo-", ""])
@pytest.mark.parametrize("swapped", [True, False])
def test_get_filename_vars(prefix, mol, press, temp, crys, swapped):
    if swapped:
        if crys is None:
            fname = f"{prefix}{mol}-T{temp}-P{press}.gsd"
        else:
            fname = f"{prefix}{mol}-T{temp}-P{press}-{crys}.gsd"
    else:
        if crys is None:
            fname = f"{prefix}{mol}-P{press}-T{temp}.gsd"
        else:
            fname = f"{prefix}{mol}-P{press}-T{temp}-{crys}.gsd"

    var = get_filename_vars(fname)
    assert isinstance(var.temperature, str)
    assert var.temperature == temp
    assert isinstance(var.pressure, str)
    assert var.pressure == press
    assert isinstance(var.crystal, type(crys))
    assert var.crystal == crys


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("dump-Trimer-P1.00-T0.40-ID1.gsd", Variables("0.40", "1.00", None, "1")),
        ("Trimer-P1.00-T0.40-ID2.gsd", Variables("0.40", "1.00", None, "2")),
        ("dump-Trimer-P1.00-T0.40-pg-ID3.gsd", Variables("0.40", "1.00", "pg", "3")),
        (
            "output/Trimer-P1.00-T0.40-ID1-p2gg.gsd",
            Variables("0.40", "1.00", "p2gg", "1"),
        ),
    ],
)
def test_filename_vars_id(filename, expected):
    variables = get_filename_vars(filename)
    assert variables == expected


@pytest.mark.parametrize("press", ["0.00", "0.50", "1.00", "13.50"])
@pytest.mark.parametrize("temp", ["0.00", "0.10", "1.50", "2.00"])
@pytest.mark.parametrize("crys", [None, "p2", "p2gg", "pg"])
@pytest.mark.parametrize("mol", ["Trimer"])
def test_set_filename_vars(mol, press, temp, crys):
    if crys is None:
        fname = f"trajectory-{mol}-P{press}-T{temp}.gsd"
    else:
        fname = f"trajectory-{mol}-P{press}-T{temp}-{crys}.gsd"

    sim_params = SimulationParams()

    set_filename_vars(fname, sim_params)
    assert isinstance(sim_params.temperature, float)
    assert sim_params.temperature == float(temp)
    assert isinstance(sim_params.pressure, float)
    assert sim_params.pressure == float(press)
    assert isinstance(sim_params.space_group, type(crys))
    assert sim_params.space_group == crys


def angle(num_elements=1):
    theta = arrays(
        np.float32,
        num_elements,
        elements=floats(
            max_value=np.finfo(np.float32).max,
            min_value=np.finfo(np.float32).min,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
    )
    return theta


def unit_quaternion_Z():
    return angle().map(lambda z: rowan.from_euler(z, 0, 0))


@pytest.fixture
def quat():
    np.random.seed(0)
    return rowan.normalize(rowan.random.rand(1000))


@given(angle())
def test_z2quaternion(angles):
    """Ensure quaternions are normalised.

    This test ensures that the quaternions returned by the z2quaternion funtion
    are normalised to a value of 1. This is important as quaternions being normalised
    is an assumption I make throughout the codebase.

    """
    result = z2quaternion(angles)
    assert np.allclose(np.linalg.norm(result, axis=1), 1)


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
    assert np.all(0 <= result)
    assert np.all(result <= 2 * np.pi)


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
    assert_allclose(result, angles, atol=1e-7)


@pytest.mark.parametrize(
    "quaternion, angle",
    [
        ([0.982_461_99, 0.0, 0.0, 0.186_462_98], 0.375_121_38),
        ([0.209_398_27, 0.0, 0.0, 0.977_830_41], 2.719_673_4),
        ([0.976_600_05, 0.0, 0.0, -0.215_063_53], -0.433_513_61),
        ([-0.211_795_45, 0.0, 0.0, 0.977_314], -2.714_769_36),
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
    result = quaternion_rotation(initial, final)
    assert np.all(result < 2 * np.pi)
    assert np.all(result > 0)


def test_quaternion_zero_rotation():
    initial = np.array([[1, 0, 0, 0]], dtype=np.float32)
    result = quaternion_rotation(initial, initial)
    assert result == 0


def test_quaternion_small_rotation():
    """Small rotations should show up as no rotation."""
    initial = np.array([[1, 0, 0, 0]])
    final = z2quaternion(np.array([1e-7]))
    result = quaternion_rotation(initial, final)
    assert result != 0
