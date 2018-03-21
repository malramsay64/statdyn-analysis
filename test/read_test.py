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

import pytest

import numpy as np
from sdanalysis import read
from sdanalysis.params import SimulationParams, paramsContext
from sdanalysis.StepSize import GenerateStepSeries

sim_params = SimulationParams(infile='test/data/trajectory-Trimer-P13.50-T3.00.gsd')


@pytest.mark.parametrize('step_limit', [0, 10, 20, 100])
def test_stopiter_handling(step_limit):
    with paramsContext(sim_params, step_limit=step_limit):
        df = read.process_file(sim_params)
    assert np.all(df.time == list(GenerateStepSeries(step_limit)))


def test_writeCache():
    with tempfile.TemporaryDirectory() as dst:
        my_list = read.WriteCache(Path(dst) / 'test1.h5')
        assert len(my_list._cache) == 0
        for i in range(100):
            my_list.append({'value': i})
        assert len(my_list._cache) == 100
        my_list.flush()
        assert len(my_list._cache) == 0


def test_writeCache_caching():
    with tempfile.TemporaryDirectory() as dst:
        my_list = read.WriteCache(Path(dst) / 'test2.h5')
        assert len(my_list._cache) == 0
        for i in range(9000):
            my_list.append({'value': i})
        assert len(my_list._cache) == 9000 - 8192
        my_list.flush()
        assert len(my_list._cache) == 0


def test_writeCache_len():
    with tempfile.TemporaryDirectory() as dst:
        my_list = read.WriteCache(Path(dst) / 'test2.h5')
        assert len(my_list._cache) == 0
        for i in range(100):
            my_list.append({'value': i})
        assert len(my_list._cache) == 100
        assert len(my_list) == 100
        for i in range(8900):
            my_list.append({'value': i})
        assert len(my_list._cache) == 9000 - 8192
        assert len(my_list) == 9000
        my_list.flush()
        assert len(my_list._cache) == 0
