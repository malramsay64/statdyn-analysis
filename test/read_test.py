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

import gsd.hoomd
import numpy as np
import pandas
import pytest

import sdanalysis
from sdanalysis import HoomdFrame, LammpsFrame, read
from sdanalysis.read import _gsd, _lammps, _read
from sdanalysis.StepSize import GenerateStepSeries


@pytest.mark.parametrize("num_steps", [0, 10, 20, 100])
def test_stopiter_handling(num_steps, infile):
    df = read.process_file(infile, wave_number=2.9, steps_max=num_steps)
    assert np.all(df.time == list(GenerateStepSeries(num_steps)))


@pytest.mark.parametrize("num_steps", [0, 10, 20, 100])
def test_linear_steps_stopiter(infile, num_steps):
    df = read.process_file(
        infile=infile, wave_number=2.9, linear_steps=None, steps_max=num_steps
    )
    assert df.time.max() == num_steps


def test_linear_sequence(infile_gsd):
    for index, _ in _gsd._gsd_linear_trajectory(infile_gsd):
        assert index == [0]


def test_linear_sequence_keyframes(infile_gsd):
    for index, frame in _gsd.read_gsd_trajectory(
        infile=infile_gsd, linear_steps=None, keyframe_interval=10
    ):
        index_len = int(np.floor(frame.timestep / 10) + 1)
        assert len(index) == index_len
        assert index == list(range(index_len))


def test_writeCache(outfile):
    my_list = _read.WriteCache(outfile)
    assert len(my_list._cache) == 0
    for i in range(100):
        my_list.append({"value": i})
    assert len(my_list._cache) == 100
    my_list.flush()
    assert len(my_list._cache) == 0


def test_writeCache_caching(outfile):
    my_list = _read.WriteCache(outfile)
    assert len(my_list._cache) == 0
    for i in range(9000):
        my_list.append({"value": i})
    assert len(my_list._cache) == 9000 - 8192
    my_list.flush()
    assert len(my_list._cache) == 0


def test_writeCache_len(outfile):
    my_list = _read.WriteCache(outfile)
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


def test_writeCache_file(outfile):
    my_list = _read.WriteCache(outfile)
    for i in range(100):
        my_list.append({"value": i})
    assert not outfile.is_file()
    my_list.flush()
    assert outfile.is_file()


def test_process_gsd(infile_gsd):
    indexes, frame = next(_gsd.read_gsd_trajectory(infile_gsd))
    assert isinstance(indexes, list)
    assert isinstance(frame, HoomdFrame)


def test_process_lammpstrj(infile_lammps):
    index, frame = next(_lammps.read_lammps_trajectory(infile_lammps))
    assert len(index) == 1
    assert index == [0]
    assert isinstance(frame, LammpsFrame)


def test_parse_lammpstrj(infile_lammps):
    num_atoms = None
    for frame in _lammps.parse_lammpstrj(infile_lammps):
        assert frame.timestep >= 0
        if num_atoms is None:
            num_atoms = len(frame)
        else:
            assert num_atoms == len(frame)
        assert np.unique(frame.position, axis=1).shape[0] == num_atoms


def test_process_file(infile):
    df = read.process_file(infile, wave_number=2.90)
    assert df is not None
    assert isinstance(df, pandas.DataFrame)


def test_process_file_outfile(infile, outfile):
    df = read.process_file(infile, wave_number=2.90, outfile=outfile)
    assert df is None

    with pandas.HDFStore(outfile) as src:
        for key in ["molecular_relaxations", "dynamics"]:
            df = src.get(key)
            assert df["temperature"].dtype == np.dtype("float64")
            assert df["pressure"].dtype == np.dtype("float64")


def test_open_trajectory(infile):
    for frame in read.open_trajectory(infile):
        # Ensure frame is of the appropriate type
        assert isinstance(frame, sdanalysis.frame.Frame)
        # Can I read the frame
        assert frame.timestep >= 0


def test_iter_trajectory(infile_gsd):
    with gsd.hoomd.open(str(infile_gsd)) as trj:
        for frame in _gsd.iter_trajectory(trj):
            assert isinstance(frame, HoomdFrame)
