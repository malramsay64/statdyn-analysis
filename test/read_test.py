#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
#
# pylint: disable=protected-access, len-as-condition
#

"""Test the statdyn.analysis.read module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas
import pytest

import sdanalysis
from sdanalysis import HoomdFrame, LammpsFrame, read
from sdanalysis.StepSize import GenerateStepSeries


@pytest.mark.parametrize("num_steps", [0, 10, 20, 100])
@pytest.mark.parametrize(
    "infile",
    [
        "test/data/trajectory-Trimer-P13.50-T3.00.gsd",
        "test/data/short-time-variance.lammpstrj",
    ],
)
def test_stopiter_handling(num_steps, infile):
    df = read.process_file(infile, wave_number=2.9, steps_max=num_steps)
    assert np.all(df.time == list(GenerateStepSeries(num_steps)))


@pytest.mark.parametrize("num_steps", [0, 10, 20, 100])
@pytest.mark.parametrize("infile", ["test/data/trajectory-Trimer-P13.50-T3.00.gsd"])
def test_linear_steps_stopiter(infile, num_steps):
    df = read.process_file(infile=infile, wave_number=2.9, steps_max=num_steps)
    assert df.time.max() == num_steps


@pytest.mark.parametrize("infile", ["test/data/trajectory-Trimer-P13.50-T3.00.gsd"])
def test_linear_sequence(infile):
    for index, _ in read._gsd_linear_trajectory(infile):
        assert index == [0]


@pytest.mark.parametrize("infile", ["test/data/trajectory-Trimer-P13.50-T3.00.gsd"])
def test_linear_sequence_keyframes(infile):
    for index, frame in read.process_gsd(
        infile=infile, linear_steps=None, keyframe_interval=10
    ):
        index_len = int(np.floor(frame.timestep / 10) + 1)
        assert len(index) == index_len
        assert index == list(range(index_len))


def test_writeCache():
    with tempfile.TemporaryDirectory() as dst:
        my_list = read.WriteCache(Path(dst) / "test1.h5")
        assert len(my_list._cache) == 0
        for i in range(100):
            my_list.append({"value": i})
        assert len(my_list._cache) == 100
        my_list.flush()
        assert len(my_list._cache) == 0


def test_writeCache_caching():
    with tempfile.TemporaryDirectory() as dst:
        my_list = read.WriteCache(Path(dst) / "test2.h5")
        assert len(my_list._cache) == 0
        for i in range(9000):
            my_list.append({"value": i})
        assert len(my_list._cache) == 9000 - 8192
        my_list.flush()
        assert len(my_list._cache) == 0


def test_writeCache_len():
    with tempfile.TemporaryDirectory() as dst:
        my_list = read.WriteCache(Path(dst) / "test2.h5")
        assert len(my_list._cache) == 0
        for i in range(100):
            my_list.append({"value": i})
        assert len(my_list._cache) == 100
        assert len(my_list) == 100
        for i in range(8900):
            my_list.append({"value": i})
        assert len(my_list._cache) == 9000 - 8192
        assert len(my_list) == 9000
        my_list.flush()
        assert len(my_list._cache) == 0


def test_writeCache_file():
    with tempfile.TemporaryDirectory() as dst:
        outfile = Path(dst) / "test2.h5"
        my_list = read.WriteCache(outfile)
        for i in range(100):
            my_list.append({"value": i})
        assert not outfile.is_file()
        my_list.flush()
        assert outfile.is_file()


@pytest.mark.parametrize("infile", ["test/data/trajectory-Trimer-P13.50-T3.00.gsd"])
def test_process_gsd(infile):
    indexes, frame = next(read.process_gsd(infile))
    assert isinstance(indexes, list)
    assert isinstance(frame, HoomdFrame)


@pytest.mark.parametrize("infile", ["test/data/short-time-variance.lammpstrj"])
def test_process_lammpstrj(infile):

    index, frame = next(read.process_lammpstrj(infile))
    assert len(index) == 1
    assert index == [0]
    assert isinstance(frame, LammpsFrame)


@pytest.mark.parametrize("infile", ["test/data/short-time-variance.lammpstrj"])
def test_parse_lammpstrj(infile):
    num_atoms = None
    for frame in read.parse_lammpstrj(infile):
        assert frame.timestep >= 0
        if num_atoms is None:
            num_atoms = len(frame)
        else:
            assert num_atoms == len(frame)
        assert np.unique(frame.position, axis=1).shape[0] == num_atoms


@pytest.mark.parametrize(
    "infile", ["short-time-variance.lammpstrj", "trajectory-Trimer-P13.50-T3.00.gsd"]
)
def test_process_file(infile):
    data_dir = Path("test/data")
    df = read.process_file(data_dir / infile, wave_number=2.90)
    assert df is not None
    assert isinstance(df, pandas.DataFrame)


@pytest.mark.parametrize(
    "infile", ["short-time-variance.lammpstrj", "trajectory-Trimer-P13.50-T3.00.gsd"]
)
def test_process_file_outfile(infile):
    data_dir = Path("test/data")
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "test.h5"
        df = read.process_file(data_dir / infile, wave_number=2.90, outfile=output)
    assert df is None


@pytest.mark.parametrize(
    "infile",
    [
        "short-time-variance.lammpstrj",
        "trajectory-Trimer-P13.50-T3.00.gsd",
        Path("trajectory-Trimer-P13.50-T3.00.gsd"),
    ],
)
def test_open_trajectory(infile):
    data_dir = Path("test/data")
    for frame in read.open_trajectory(data_dir / infile):
        # Ensure frame is of the appropriate type
        assert isinstance(frame, sdanalysis.frame.Frame)
        # Can I read the frame
        assert frame.timestep >= 0
