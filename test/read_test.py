#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Test the statdyn.analysis.read module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from sdanalysis import read
from sdanalysis.params import SimulationParams
from sdanalysis.StepSize import GenerateStepSeries


@pytest.fixture
def sim_params():
    yield SimulationParams(infile="test/data/trajectory-Trimer-P13.50-T3.00.gsd")


@pytest.mark.parametrize("num_steps", [0, 10, 20, 100])
@pytest.mark.parametrize(
    "infile",
    [
        "test/data/trajectory-Trimer-P13.50-T3.00.gsd",
        "test/data/short-time-variance.lammpstrj",
    ],
)
def test_stopiter_handling(sim_params, num_steps, infile):
    with sim_params.temp_context(infile=infile, num_steps=num_steps):
        df = read.process_file(sim_params)
    assert np.all(df.time == list(GenerateStepSeries(num_steps)))


@pytest.mark.parametrize("num_steps", [0, 10, 20, 100])
def test_linear_steps_stopiter(sim_params, num_steps):
    with sim_params.temp_context(num_steps=num_steps, linear_steps=None):
        df = read.process_file(sim_params)
    assert df.time.max() == num_steps


def test_linear_sequence(sim_params):
    with sim_params.temp_context(linear_steps=None):
        for index, _ in read.process_gsd(sim_params):
            assert index == [0]


def test_linear_sequence_keyframes(sim_params):
    with sim_params.temp_context(linear_steps=None, gen_steps=10):
        for index, frame in read.process_gsd(sim_params):
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


def test_process_gsd(sim_params):
    indexes, frame = next(read.process_gsd(sim_params))
    assert isinstance(indexes, list)
    assert isinstance(frame, read.HoomdFrame)


def test_process_lammpstrj():
    pass


def test_process_file():
    pass
