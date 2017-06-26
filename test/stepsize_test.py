#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Testing for the stepsize module."""

import pytest
from statdyn.StepSize import GenerateStepSeries, generate_steps


@pytest.fixture(params=[
    {'max': 100, 'lin': 10, 'start': 0,
     'def':
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
    {'max': 99, 'lin': 10, 'start': 0,
     'def':
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]},
    {'max': 87, 'lin': 10, 'start': 0,
     'def':
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 87]},
    {'max': 10, 'lin': 5, 'start': -5,
     'def':
     [-4, -3, -2, -1, 0, 10]},
])
def steps(request):
    """Generate the steps lists."""
    request.param['gen'] = list(generate_steps(request.param['max'],
                                               request.param['lin'],
                                               request.param['start']))
    return request.param


def test_generate_steps(steps):  # pylint: disable=redefined-outer-name
    """Test generation of steps."""
    assert steps['gen'][-1] == steps['max']
    assert steps['gen'] == steps['def']


def test_generate_step_series():
    """Test generate_steps and generate_step_series.

    This ensures that both functions give same results for case of only a
    single series
    """
    single = list(generate_steps(1000, 10, 0))
    series = list(GenerateStepSeries(1000, 10, 10000, 1))
    print(series)
    assert single == series


def test_generate_step_series_many():
    """Test generation of a step series works."""
    list(GenerateStepSeries(10000, 10, 1000, 100))


def test_next():
    """Ensure next() works the same as __iter__."""
    list_iter = list(GenerateStepSeries(1000, 10, 0))
    next_iter = []
    gen = GenerateStepSeries(1000, 10, 0)
    try:
        while True:
            next_iter.append(gen.next())
    except StopIteration:
        pass
    assert list_iter == next_iter
