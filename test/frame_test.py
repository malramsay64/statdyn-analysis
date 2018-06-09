#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
import gsd.hoomd
import numpy as np
import pytest
from sdanalysis.frame import gsdFrame, lammpsFrame


@pytest.fixture
def lammps_frame():
    inframe = {
        "x": np.random.rand(100),
        "y": np.random.rand(100),
        "z": np.random.rand(100),
        "box": [1, 1, 1],
        "timestep": 0,
    }
    frame = lammpsFrame(inframe)
    return frame


@pytest.fixture
def gsd_frame():
    with gsd.hoomd.open("test/data/trajectory-Trimer-P13.50-T3.00.gsd") as trj:
        inframe = trj[0]
    frame = gsdFrame(inframe)
    return frame


@pytest.fixture(
    params=[pytest.lazy_fixture("lammps_frame"), pytest.lazy_fixture("gsd_frame")],
    ids=["lammps_frame", "gsd_frame"],
)
def frametypes(request):
    return request.param


def test_frame_len(frametypes):
    assert len(frametypes) > 0
    assert len(frametypes) == len(frametypes.position)


def test_frame_position(frametypes):
    assert len(frametypes.position) == len(frametypes)
    assert np.all(frametypes.position == frametypes.position)
    assert frametypes.position.shape == (len(frametypes), 3)
    assert frametypes.position.dtype == np.float32


def test_frame_orientation(frametypes):
    assert len(frametypes.orientation) == len(frametypes)
    assert np.all(frametypes.orientation == frametypes.orientation)
    assert frametypes.orientation.shape == (len(frametypes), 4)
    assert frametypes.orientation.dtype == np.float32


def test_frame_timestep(frametypes):
    frametypes.timestep


def test_frame_box(frametypes):
    assert len(frametypes.box) >= 3
    assert np.all(frametypes.box >= 0)
    assert frametypes.box.dtype == np.float32


@pytest.fixture(
    params=[0, 1, 2, 3],
    ids=["particles == mols", "particles == 3*mols", "body == 2^32-1", "body == None"],
)
def gsd_test_frames(request):
    num_mols = 100
    snap = gsd.hoomd.Snapshot()
    snap.particles.N = num_mols
    if request.param == 1:
        snap.particles.N = 3 * num_mols

    body_vals = {
        0: np.arange(num_mols, dtype=np.uint32),
        1: np.tile(np.arange(num_mols, dtype=np.uint32), 3),
        2: np.full(num_mols, 2 ** 32 - 1, dtype=np.uint32),
        3: None,
    }
    snap.particles.body = body_vals.get(request.param)
    snap.particles.position = np.zeros((snap.particles.N, 3))
    return num_mols, snap


def test_gsd_num_bodies(gsd_test_frames):
    num_mols, snap = gsd_test_frames
    assert gsdFrame._get_num_bodies(snap) == num_mols
