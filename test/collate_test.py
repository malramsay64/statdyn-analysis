#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test correctness of collate functions."""

import numpy as np
import pytest

from statdyn.datagen.collate import spearman_rank

SPEARMAN = [
    (np.arange(10), np.arange(10), -1.),
    (np.arange(10), -np.arange(10), 1.),
    (np.arange(10), 10 - np.arange(10), 1.),
]


@pytest.mark.parametrize('trans, rot, result', SPEARMAN)
@pytest.mark.xfail
def test_spearman(trans, rot, result):
    """Simple test for the spearman rank correlation."""
    assert np.isclose(spearman_rank(trans, rot, fraction=1), result)
