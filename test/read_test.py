#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the statdyn.analysis.read module."""

import numpy as np
import pytest

from sdanalysis import read
from sdanalysis.StepSize import GenerateStepSeries


@pytest.mark.parametrize('step_limit', [0, 10, 20, 100])
def test_stopiter_handling(step_limit):
    df = read.process_gsd('test/data/trajectory-13.50-3.00.gsd', step_limit=step_limit)
    assert np.all(df.time == list(GenerateStepSeries(step_limit)))
