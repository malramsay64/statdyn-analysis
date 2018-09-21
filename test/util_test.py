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
import pytest

from sdanalysis.params import SimulationParams
from sdanalysis.util import get_filename_vars, orientation2positions, set_filename_vars


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
@pytest.mark.parametrize("mol", ["Trimer"])
def test_get_filename_vars(mol, press, temp):
    fname = f"trajectory-{mol}-P{press}-T{temp}.gsd"
    var = get_filename_vars(fname)
    assert isinstance(var.temperature, str)
    assert var.temperature == temp
    assert isinstance(var.pressure, str)
    assert var.pressure == press


@pytest.mark.parametrize("press", ["0.00", "0.50", "1.00", "13.50"])
@pytest.mark.parametrize("temp", ["0.00", "0.10", "1.50", "2.00"])
@pytest.mark.parametrize("mol", ["Trimer"])
def test_set_filename_vars(mol, press, temp):
    fname = f"trajectory-{mol}-P{press}-T{temp}.gsd"
    sim_params = SimulationParams()

    set_filename_vars(fname, sim_params)
    assert isinstance(sim_params.temperature, float)
    assert sim_params.temperature == float(temp)
    assert isinstance(sim_params.pressure, float)
    assert sim_params.pressure == float(press)
