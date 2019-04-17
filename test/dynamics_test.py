#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
#
# pylint: disable=redefined-outer-name, protected-access, no-self-use
#

"""Testing the dynamics module."""

import gsd.hoomd
import numpy as np
import pytest
from freud.box import Box
from hypothesis import assume, example, given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from numpy.testing import assert_allclose

from sdanalysis import HoomdFrame, dynamics
from sdanalysis.read import process_gsd

MAX_BOX = 20.0
DTYPE = np.float32
EPS = 2 * np.sqrt(np.finfo(DTYPE).eps)
HYP_DTYPE = DTYPE


def test_calculate_max_wavenumber(wavenumber=10):
    angles = np.linspace(0, 2 * np.pi, num=6, endpoint=False).reshape((-1, 1))
    radial = np.concatenate(
        [np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1
    )
    positions = []
    for i in range(1, 20):
        positions.append(radial + i * (2 * np.pi / wavenumber))
    positions = np.concatenate(positions)
    print(positions)

    box = Box(Lx=100, Ly=100, is2D=True)

    calc_wavenumber = dynamics._calculate_wave_number(box, positions)

    assert calc_wavenumber >= 0


def translational_displacement_reference(
    box: Box, initial: np.ndarray, final: np.ndarray
) -> np.ndarray:
    """Simplified reference implementation for computing the displacement.

    This computes the displacement using the shortest path from the original
    position to the final position.

    """
    result = np.empty(final.shape[0], dtype=final.dtype)
    box = np.array([box.Lx, box.Ly, box.Lz])
    for index, _ in enumerate(result):
        temp = initial[index] - final[index]
        for i in range(3):
            if temp[i] > box[i] / 2:
                temp[i] -= box[i]
            if temp[i] < -box[i] / 2:
                temp[i] += box[i]
        result[index] = np.linalg.norm(temp, axis=0)
    return result


@given(
    arrays(HYP_DTYPE, (10, 3), elements=floats(-MAX_BOX / 4, MAX_BOX / 4)),
    arrays(HYP_DTYPE, (10, 3), elements=floats(-MAX_BOX / 4, MAX_BOX / 4)),
)
def test_translational_displacement_noperiod(init, final):
    """Test calculation of the translational displacement.

    This test ensures that the result is close to the func::`numpy.linalg.norm`
    function in the case where there is no periodic boundaries to worry
    about.
    """
    box = Box(MAX_BOX, MAX_BOX, MAX_BOX)
    np_res = np.linalg.norm(init - final, axis=1)
    result = dynamics.translational_displacement(box, init, final)
    ref_res = translational_displacement_reference(box, init, final)
    print(result)
    assert_allclose(result, np_res, atol=EPS)
    assert_allclose(result, ref_res, atol=EPS)


@given(
    arrays(HYP_DTYPE, (10, 3), elements=floats(-MAX_BOX / 2, -MAX_BOX / 4 - 1e-5)),
    arrays(HYP_DTYPE, (10, 3), elements=floats(MAX_BOX / 4, MAX_BOX / 2)),
)
def test_translational_displacement_periodicity(init, final):
    """Ensure the periodicity is calculated appropriately.

    This is testing that periodic boundaries are identified appropriately.
    """
    box = Box(MAX_BOX, MAX_BOX, MAX_BOX)
    np_res = np.square(np.linalg.norm(init - final, axis=1))
    result = dynamics.translational_displacement(box, init, final)
    ref_res = translational_displacement_reference(box, init, final)
    assert np.all(np.logical_not(np.isclose(result, np_res)))
    assert_allclose(result, ref_res, atol=EPS)


@given(
    arrays(HYP_DTYPE, (10, 3), elements=floats(-MAX_BOX / 2, MAX_BOX / 2)),
    arrays(HYP_DTYPE, (10, 3), elements=floats(-MAX_BOX / 2, MAX_BOX / 2)),
)
def test_translational_displacement(init, final):
    """Ensure the periodicity is calculated appropriately.

    This is testing that periodic boundaries are identified appropriately.
    """
    box = Box(MAX_BOX, MAX_BOX, MAX_BOX)
    result = dynamics.translational_displacement(box, init, final)
    ref_res = translational_displacement_reference(box, init, final)
    assert np.allclose(result, ref_res, atol=EPS)


@given(arrays(HYP_DTYPE, (100), elements=floats(0, 10)))
def test_alpha(displacement):
    """Test the computation of the non-gaussian parameter."""
    alpha = dynamics.alpha_non_gaussian(displacement)
    assume(not np.isnan(alpha))
    assert alpha >= -1


@given(
    arrays(HYP_DTYPE, (100), elements=floats(0, 10)),
    arrays(HYP_DTYPE, (100), elements=floats(0, 2 * np.pi)),
)
def test_overlap(displacement, rotation):
    """Test the computation of the overlap of the largest values."""
    overlap_same = dynamics.mobile_overlap(rotation, rotation)
    assert np.isclose(overlap_same, 1)
    overlap = dynamics.mobile_overlap(displacement, rotation)
    assert 0.0 <= overlap <= 1.0


@pytest.fixture(scope="module")
def trajectory():
    with gsd.hoomd.open("test/data/trajectory-Trimer-P13.50-T3.00.gsd") as trj:
        yield trj


@pytest.fixture(scope="module")
def dynamics_class(trajectory):
    snap = HoomdFrame(trajectory[0])
    return dynamics.Dynamics(
        snap.timestep,
        snap.box,
        snap.position,
        snap.orientation,
        image=snap.image,
        wave_number=4.0,
    )


class TestDynamicsClass:
    @pytest.mark.parametrize("step", [0, 1, 10, 20])
    def test_displacements(self, dynamics_class, trajectory, step):
        snap = HoomdFrame(trajectory[step])
        displacement = dynamics_class.get_displacements(snap.position)
        assert displacement.shape == (dynamics_class.num_particles,)
        if step == 0:
            assert np.all(displacement == 0.0)
        else:
            assert np.all(displacement >= 0.0)

    @pytest.mark.parametrize("step", [0, 1, 10, 20])
    def test_displacements_image(self, dynamics_class, trajectory, step):
        snap = HoomdFrame(trajectory[step])
        displacement = dynamics_class.get_displacements(snap.position, snap.image)
        assert displacement.shape == (dynamics_class.num_particles,)
        if step == 0:
            assert np.all(displacement == 0.0)
        else:
            assert np.all(displacement >= 0.0)
            assert np.max(displacement) <= 0.3

    @pytest.mark.parametrize("step", [0, 1, 10, 20])
    def test_image(self, dynamics_class, trajectory, step):
        snap = HoomdFrame(trajectory[step])
        displacement = dynamics_class.get_displacements(snap.position, snap.image)
        assert displacement.shape == (dynamics_class.num_particles,)
        assert np.max(np.abs(dynamics_class.image - snap.image)) <= 1

    @pytest.mark.parametrize("step", [0, 1, 10, 20])
    @pytest.mark.parametrize("method", ["compute_msd", "compute_mfd"])
    def test_trans_methods(self, dynamics_class, trajectory, step, method):
        snap = trajectory[step]
        quantity = getattr(dynamics_class, method)(snap.particles.position)

        if step == 0:
            assert np.isclose(quantity, 0, atol=1e-7)
        else:
            assert quantity >= 0

    @pytest.mark.parametrize("step", [0, 1, 10, 20])
    @pytest.mark.parametrize("method", ["compute_alpha"])
    def test_alpha_methods(self, dynamics_class, trajectory, step, method):
        snap = trajectory[step]
        quantity = getattr(dynamics_class, method)(snap.particles.position)
        assert isinstance(quantity, float)

    @pytest.mark.parametrize("step", [0, 1, 10, 20])
    @pytest.mark.parametrize("method", ["compute_rotation"])
    def test_rot_methods(self, dynamics_class, trajectory, step, method):
        snap = trajectory[step]
        quantity = getattr(dynamics_class, method)(snap.particles.orientation)

        if step == 0:
            assert np.isclose(quantity, 0, atol=1e-7)
        else:
            assert quantity >= 0

    @pytest.mark.parametrize("step", [0, 1, 10, 20])
    def test_rotations(self, dynamics_class, trajectory, step):
        snap = trajectory[step]
        rotations = dynamics_class.get_rotations(snap.particles.orientation)
        assert rotations.shape == (dynamics_class.num_particles,)
        if step == 0:
            assert np.allclose(rotations, 0.0, atol=EPS)
        else:
            assert np.all(rotations >= 0.0)

    def test_float64_box(self):
        box = Box.cube(1)
        init = np.random.random((100, 3)).astype(np.float32)
        final = np.random.random((100, 3)).astype(np.float32)
        result = dynamics.translational_displacement(box, init, final)
        assert np.all(result < 1)

    def test_read_only_arrays(self):
        box = Box.cube(1)
        init = np.random.random((100, 3)).astype(np.float32)
        init.flags.writeable = False
        final = np.random.random((100, 3)).astype(np.float32)
        final.flags.writeable = False
        result = dynamics.translational_displacement(box, init, final)
        assert np.all(result < 1)


def test_process_file():
    process_gsd("test/data/trajectory-Trimer-P13.50-T3.00.gsd")


def test_MolecularRelaxation():
    num_elements = 10
    threshold = 0.4
    tau = dynamics.MolecularRelaxation(num_elements, threshold)
    invalid_values = np.full(num_elements, tau._max_value, dtype=np.uint32)

    def move(dist):
        return np.ones(num_elements) * dist

    # No motion
    tau.add(1, move(0))
    assert np.all(tau.get_status() == invalid_values)
    # Small motion inside threshold
    tau.add(2, move(threshold - 0.1))
    assert np.all(tau.get_status() == invalid_values)
    # Move outside threshold
    tau.add(3, move(threshold + 0.1))
    assert np.all(tau.get_status() == np.full(num_elements, 3))
    # Move inside threshold
    tau.add(4, move(threshold - 0.1))
    assert np.all(tau.get_status() == np.full(num_elements, 3))
    # Move outside threshold again
    tau.add(4, move(threshold + 0.1))
    assert np.all(tau.get_status() == np.full(num_elements, 3))


def test_LastMolecularRelaxation():
    num_elements = 10
    threshold = 0.4
    tau = dynamics.LastMolecularRelaxation(num_elements, threshold, 1.0)
    invalid_values = np.full(num_elements, tau._max_value, dtype=np.uint32)

    def move(dist):
        return np.ones(num_elements) * dist

    # Move past threshold
    tau.add(2, move(threshold + 0.1))
    assert np.all(tau.get_status() == invalid_values)
    assert np.all(tau._status == np.full(num_elements, 2))
    assert np.all(tau._state == np.ones(num_elements, dtype=np.uint8))
    # Move inside threshold
    tau.add(3, move(threshold - 0.1))
    assert np.all(tau.get_status() == invalid_values)
    assert np.all(tau._status == np.full(num_elements, 2))
    assert np.all(tau._state == np.zeros(num_elements, dtype=np.uint8))
    # Move outside threshold again
    tau.add(4, move(threshold + 0.1))
    assert np.all(tau.get_status() == invalid_values)
    assert np.all(tau._status == np.full(num_elements, 4))
    assert np.all(tau._state == np.ones(num_elements, dtype=np.uint8))
    # Move outside threshold again
    tau.add(5, move(threshold + 0.2))
    assert np.all(tau.get_status() == invalid_values)
    assert np.all(tau._status == np.full(num_elements, 4))
    # Move past irreversibility
    tau.add(6, move(1.1))
    assert np.all(tau.get_status() == np.full(num_elements, 4))
    assert np.all(tau._status == np.full(num_elements, 4))
    assert np.all(
        tau._state == np.ones(num_elements, dtype=np.uint8) * tau._is_irreversible
    )
    # Move inside threshold
    tau.add(7, move(threshold - 0.1))
    assert np.all(tau.get_status() == np.full(num_elements, 4))
    assert np.all(tau._status == np.full(num_elements, 4))
    assert np.all(
        tau._state == np.ones(num_elements, dtype=np.uint8) * tau._is_irreversible
    )
    # Move outside threshold, shouldn't update
    tau.add(8, move(threshold + 0.1))
    assert np.all(tau.get_status() == np.full(num_elements, 4))
    assert np.all(tau._status == np.full(num_elements, 4))
    assert np.all(
        tau._state == np.ones(num_elements, dtype=np.uint8) * tau._is_irreversible
    )


@given(arrays(HYP_DTYPE, (10), elements=floats(0, 1)))
@example(np.full(10, np.nan))
def test_structural_relaxation(array):
    value = dynamics.structural_relax(array)
    assert isinstance(value, float)
