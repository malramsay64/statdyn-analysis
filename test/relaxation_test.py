#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the relaxation module."""

from pathlib import Path

import numpy as np
import pytest
from sdanalysis import relaxation


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
    time = np.arange(101)

    def test_compute(self):
        value = 50 - np.abs(np.arange(-50, 50))
        max_value, error = relaxation.max_value_relaxation(self.time, value)
        assert max_value == 50
        assert error == 1

    def test_compute_nan(self):
        value = 50. - np.abs(np.arange(-50, 50))
        value[0] = np.nan
        max_value, error = relaxation.max_value_relaxation(self.time, value)
        assert max_value == 50
        assert error == 1

    @pytest.mark.parametrize("expected_max", ["first", "last"])
    def test_boundary_values(self, expected_max):
        """The first value should work when the maximum."""
        value = np.zeros_like(self.time)
        if expected_max == "first":
            value[0] = 1
        elif expected_max == "last":
            value[-1] = 1
        max_value, error = relaxation.max_value_relaxation(self.time, value)
        assert max_value == 1
        assert error == 1


class TestMaxTimeRelax:
    time = np.arange(101)

    def test_compute(self):
        value = 50 - np.abs(np.arange(-50, 50))
        max_time, error = relaxation.max_time_relaxation(self.time, value)
        assert max_time == 50
        assert error == 1

    def test_compute_nan(self):
        value = 50. - np.abs(np.arange(-50, 50))
        value[0] = np.nan
        max_time, error = relaxation.max_time_relaxation(self.time, value)
        assert max_time == 50
        assert error == 1

    @pytest.mark.parametrize("expected_max", ["first", "last"])
    def test_boundary_values(self, expected_max):
        """The first value should work when the maximum."""
        value = np.zeros_like(self.time)
        if expected_max == "first":
            value[0] = 1
            expected_time = 0
        elif expected_max == "last":
            value[-1] = 1
            expected_time = self.time[-1]
        max_time, error = relaxation.max_time_relaxation(self.time, value)
        assert max_time == expected_time
        assert error == 1


def test_compute_relaxations():
    infile = Path("test/data/dynamics.h5")
    relaxation.compute_relaxations(infile)
