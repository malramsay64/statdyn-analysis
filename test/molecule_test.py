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

MOLECULE_LIST = [
    molecules.Molecule,
    molecules.Trimer,
    molecules.Dimer,
    molecules.Disc,
    molecules.Sphere,
]


@pytest.fixture(scope='module', params=MOLECULE_LIST)
def mol(request):
    return request.param()


def test_compute_moment_inertia(mol):
    mom_I = np.array(mol.moment_inertia)
    assert np.all(mom_I[:2] == 0)


def test_scale_moment_inertia(mol):
    scale_factor = 10.
    init_mom_I = np.array(mol.moment_inertia)
    mol.scale_moment_inertia(scale_factor)
    final_mom_I = np.array(mol.moment_inertia)
    assert np.all(scale_factor*init_mom_I == final_mom_I)


def test_get_radii(mol):
    radii = mol.get_radii()
    assert radii[0] == 1.


def test_read_only_position(mol):
    assert mol.positions.flags.writeable == False


def test_orientation2positions(mol):
    position = np.array([[0, 0, 0]], dtype=np.float32)
    orientation = np.array([[1, 0, 0, 0]], dtype=np.float32)
    x_inv_pos = np.copy(mol.positions)
    x_inv_pos[:, 0] = -x_inv_pos[:, 0]
    rotated_pos = mol.orientation2positions(position, orientation)
    assert np.allclose(
        rotated_pos,
        x_inv_pos,
        atol=1e5,
    )


def test_orientation2positions_invert_xy(mol):
    position = np.array([[0, 0, 0]], dtype=np.float32)
    orientation = np.array([[0, 0, 0, 1]], dtype=np.float32)
    xy_inv_pos = np.copy(mol.positions)
    xy_inv_pos[:, :2] = -xy_inv_pos[:, :2]
    rotated_pos = mol.orientation2positions(position, orientation)
    assert np.allclose(
        rotated_pos,
        xy_inv_pos,
        atol=1e5,
    )


def test_orientation2positions_moved(mol):
    position = np.array([[1, 1, 0]], dtype=np.float32)
    orientation = np.array([[1, 0, 0, 0]], dtype=np.float32)
    rotated_pos = mol.orientation2positions(position, orientation)
    moved_pos = mol.positions + np.repeat(position, mol.num_particles, axis=0)
    assert np.allclose(
        rotated_pos,
        moved_pos,
    )


def test_orientation2positions_moved_rot(mol):
    position = np.array([[4, 2, 0]], dtype=np.float32)
    orientation = np.array([[0, 0, 0, 1]], dtype=np.float32)
    rotated_pos = mol.orientation2positions(position, orientation)
    xy_inv_pos = np.copy(mol.positions)
    xy_inv_pos[:, :2] = -xy_inv_pos[:, :2]
    moved_pos = xy_inv_pos + np.tile(position, (mol.num_particles, 1))
    assert np.allclose(
        rotated_pos,
        moved_pos,
    )


def test_orientation2positions_moved_rot_multiple(mol):
    position = np.array([[4, 2, 0], [0, 0, 0]], dtype=np.float32)
    orientation = np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32)
    rotated_pos = mol.orientation2positions(position, orientation)
    xy_inv_pos = np.copy(mol.positions)
    xy_inv_pos[:, :2] = -xy_inv_pos[:, :2]
    moved_pos = (np.repeat(xy_inv_pos, position.shape[0], axis=0)
                 + np.tile(position, (mol.num_particles, 1)))
    assert np.allclose(
        rotated_pos,
        moved_pos,
    )


def test_get_types(mol):
    mol.get_types()


def test_moment_inertia_trimer():
    """Ensure calculation of moment of inertia is working properly."""
    mol = molecules.Trimer()
    assert mol.moment_inertia == (0, 0, 1.6666666666666665)
    mol = molecules.Trimer(distance=0.8)
    assert mol.moment_inertia[2] < 1.6666666666666665
    mol = molecules.Trimer(distance=1.2)
    assert mol.moment_inertia[2] > 1.6666666666666665


@given(floats(min_value=0, allow_infinity=False, allow_nan=False))
def test_moment_inertia_scaling(scaling_factor):
    """Test that the scaling factor is working properly."""
    reference = molecules.Trimer()
    with np.errstate(over='ignore'):
        scaled = molecules.Trimer(moment_inertia_scale=scaling_factor)
        assert len(reference.moment_inertia) == len(scaled.moment_inertia)
        assert np.allclose(np.array(reference.moment_inertia)*scaling_factor,
                           np.array(scaled.moment_inertia))


def test_compute_size(mol):
    size = mol.compute_size()
    assert size >= 2.
