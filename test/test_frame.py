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
        'x': np.random.rand(100),
        'y': np.random.rand(100),
        'z': np.random.rand(100),
        'box': [1, 1, 1],
        'timestep': 0,
    }
    frame = lammpsFrame(inframe)
    return frame


@pytest.fixture
def gsd_frame():
    inframe = gsd.hoomd.open('test/data/Trimer-13.50-3.00.gsd')[0]
    frame = gsdFrame(inframe)
    return frame


@pytest.fixture(
    params=[pytest.lazy_fixture('lammps_frame'), pytest.lazy_fixture('gsd_frame')]
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
