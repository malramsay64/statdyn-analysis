#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the relaxation module."""

import numpy as np

from statdyn.analysis import relaxation


def test_diffusion_constant():
    """Ensure the diffusion constant is giving a reasonable result."""
    known_diffusion = 1e-3
    offset = 1e-4
    time = np.arange(10000)
    msd = time*known_diffusion/4 + offset
    diff, diff_err = relaxation.diffusion_constant(time, msd)
    assert np.isclose(diff, known_diffusion)
    assert np.isclose(diff_err, 0)


def test_exponential_relax():
    """Ensure the exponential_relaxation functiton working apropriately."""
    known_decay = 1e4
    time = np.arange(100000)
    value = np.exp(-time/known_decay)
    relax, *_ = relaxation.exponential_relaxation(time, value)
    assert np.isclose(relax, known_decay)
