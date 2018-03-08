#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
from sdanalysis import frame
import pytest
import pandas
import gsd.hoomd
import numpy as np


@pytest.fixture
def lammps_frame():
    inframe = pandas.DataFrame(
        {'x': np.random.rand(100), 'y': np.random.rand(100), 'z': np.random.rand(100)}
    )
    box = [1, 1, 1]
    return frame.lammpsFrame(0, box, inframe)


@pytest.fixture
def gsd_frame():
    inframe = gsd.hoomd.open('test/data/Trimer-13.50-3.00.gsd')[0]
    return frame.gsdFrame(inframe)


@pytest.fixture(
    params=[pytest.mark.fixture(lammps_frame), pytest.mark.fixture(gsd_frame)]
)
def frametypes(request):
    return request.param()


def test_frame_len(frametypes):
    assert len(frametypes) > 0


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
