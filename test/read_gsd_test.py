#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

from pathlib import Path
from tempfile import TemporaryDirectory

import gsd.hoomd
import numpy as np
import pytest

from sdanalysis import HoomdFrame
from sdanalysis.read._gsd import read_gsd_trajectory
from sdanalysis.StepSize import GenerateStepSeries


@pytest.fixture()
def gsd_trajectory():
    with TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "test.gsd"
        step_iter = GenerateStepSeries(
            total_steps=10000, num_linear=100, gen_steps=1000
        )
        with gsd.hoomd.open(str(tmp_file), "wb") as dst:
            for timestep in step_iter:
                snap = gsd.hoomd.Snapshot()

                snap.configuration.step = timestep
                snap.configuration.dimensions = 2
                snap.configuration.box = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
                snap.particles.N = 10
                snap.types = ["A"]
                snap.position = np.zeros((10, 3))
                snap.typeid = np.zeros(10, dtype=np.uint32)
                dst.append(snap)
        yield tmp_file


def test_traj(gsd_trajectory):
    with gsd.hoomd.open(str(gsd_trajectory)) as trj:
        HoomdFrame(trj[0])


def test_read_trajectory_linear(gsd_trajectory):
    traj_iterator = read_gsd_trajectory(
        gsd_trajectory, steps_max=100, linear_steps=None
    )
    for index, (_, snap) in enumerate(traj_iterator):
        assert isinstance(snap, HoomdFrame)
        assert snap.timestep == index


def test_read_trajectory_exp(gsd_trajectory):
    traj_iterator = read_gsd_trajectory(
        gsd_trajectory, steps_max=200, linear_steps=100, keyframe_interval=1000
    )
    step_iter = GenerateStepSeries(total_steps=200, num_linear=100, gen_steps=1000)
    for index, (_, snap) in zip(step_iter, traj_iterator):
        assert isinstance(snap, HoomdFrame)
        assert snap.timestep == index
