#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the relaxation module."""


import numpy as np
import pandas
import pytest
from hypothesis import example, given
from hypothesis.extra.numpy import arrays

from sdanalysis import relaxation

relaxation_types = ["alpha", "gamma", "com_struct", "msd", "rot1", "rot2", "struct"]


def test_diffusion_constant():
    """Ensure the diffusion constant is giving a reasonable result."""
    known_diffusion = 1e-3
    offset = 1e-4
    time = np.arange(10000)
    msd = time * known_diffusion + offset
    diff, diff_err = relaxation.diffusion_constant(time, msd)
    assert np.isclose(diff, known_diffusion)
    assert np.isclose(diff_err, 0)


def test_exponential_relax():
    """Ensure the exponential_relaxation functiton working apropriately."""
    known_decay = 1e4
    time = np.arange(100000)
    value = np.exp(-time / known_decay)
    relax, *_ = relaxation.exponential_relaxation(time, value)
    assert np.isclose(relax, known_decay)


def test_threshold_relaxation():
    num_values = 1000
    time = np.arange(num_values)
    values = np.arange(num_values) / num_values
    relax, _ = relaxation.threshold_relaxation(
        time, values, threshold=0.5, greater=True
    )
    assert relax == num_values / 2 + 1
    relax, _ = relaxation.threshold_relaxation(
        time, values, threshold=0.5, greater=False
    )
    assert relax == num_values / 2


class TestMaxValueRelax:
    time = np.arange(100)

    def test_compute(self):
        value = 50 - np.abs(np.arange(-50, 50))
        assert self.time.shape == value.shape
        max_value, error = relaxation.max_value_relaxation(self.time, value)
        assert max_value == 50
        assert error == 1

    def test_compute_nan(self):
        value = 50. - np.abs(np.arange(-50, 50))
        assert self.time.shape == value.shape
        value[0] = np.nan
        max_value, error = relaxation.max_value_relaxation(self.time, value)
        assert max_value == 50
        assert error == 1

    @pytest.mark.parametrize("expected_max", ["first", "last"])
    def test_boundary_values(self, expected_max):
        """The first value should work when the maximum."""
        value = np.zeros_like(self.time)
        assert self.time.shape == value.shape
        if expected_max == "first":
            value[0] = 1
        elif expected_max == "last":
            value[-1] = 1
        max_value, error = relaxation.max_value_relaxation(self.time, value)
        assert max_value == 1
        assert error == 1


class TestMaxTimeRelax:
    time = np.arange(100)

    def test_compute(self):
        value = 50 - np.abs(np.arange(-50, 50))
        assert self.time.shape == value.shape
        max_time, error = relaxation.max_time_relaxation(self.time, value)
        assert max_time == 50
        assert error == 1

    def test_compute_nan(self):
        value = 50. - np.abs(np.arange(-50, 50))
        assert self.time.shape == value.shape
        value[0] = np.nan
        max_time, error = relaxation.max_time_relaxation(self.time, value)
        assert max_time == 50
        assert error == 1

    @pytest.mark.parametrize("expected_max", ["first", "last"])
    def test_boundary_values(self, expected_max):
        """The first value should work when the maximum."""
        value = np.zeros_like(self.time)
        assert self.time.shape == value.shape
        if expected_max == "first":
            value[0] = 1
            expected_time = 0
        elif expected_max == "last":
            value[-1] = 1
            expected_time = self.time[-1]
        max_time, error = relaxation.max_time_relaxation(self.time, value)
        assert max_time == expected_time
        assert error == 1


def test_compute_relaxations(dynamics_file):
    relaxation.compute_relaxations(dynamics_file)


def test_compute_relaxations_values(dynamics_file):
    relaxation.compute_relaxations(dynamics_file)
    df = pandas.read_hdf(dynamics_file, "relaxations")
    columns = ["diffusion_constant"]
    for col in columns:
        assert df[col].dtype == float


@given(values=arrays(dtype=np.float32, shape=1000))
@example(values=np.ones(1000))
@pytest.mark.parametrize("relax_type", relaxation_types)
def test_compute_relaxations_random(values, relax_type):
    timesteps = np.arange(values.shape[0], dtype=int)
    relaxation.compute_relaxation_value(timesteps, values, relax_type)


@given(values=arrays(dtype=np.float32, shape=1000))
@pytest.mark.parametrize("relax_type", relaxation_types)
def test_series_relaxations_random(values, relax_type):
    timesteps = np.arange(values.shape[0], dtype=int)
    s = pandas.Series(data=values, index=timesteps, name=relax_type)
    relaxation.series_relaxation_value(s)


@pytest.mark.parametrize(
    "indexes", [["temperature"], ["pressure"], ["temperature", "pressure"]]
)
def test_series_relaxations_index(indexes):
    array_size = 1000
    data = {i: 1. for i in indexes}
    data["timesteps"] = np.arange(array_size)
    data["test"] = np.ones(array_size)
    df = pandas.DataFrame(data).groupby(indexes).mean()
    s = df.test
    relaxation.series_relaxation_value(s)


@pytest.fixture
def relax_df():
    # Use one frame to detect when there is incorrect handling of the harmonic mean
    frames = 1
    mols = 100
    np.random.seed(0)
    df = pandas.DataFrame(
        {
            "init_frame": np.repeat(np.arange(frames), mols),
            "molecule": np.tile(np.arange(mols), frames),
            "temperature": 0.1,
            "pressure": 0.1,
            "tau_DL04": np.random.random(frames * mols),
            "tau_D04": np.random.random(frames * mols),
        }
    )
    return df.groupby(["init_frame", "molecule"]).mean()


def test_relaxation_hmean(relax_df):
    df = relaxation.compute_molecular_relaxations(relax_df)
    columns = ["tau_DL04", "tau_D04"]
    for col in columns:
        assert np.all(df[col + "_mean"].values != df[col + "_hmean"].values)
