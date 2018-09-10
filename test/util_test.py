#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""
import numpy as np

from sdanalysis.util import orientation2positions


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
