#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Testing for the stepsize module."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers
from sdanalysis.StepSize import GenerateStepSeries, generate_steps


@pytest.fixture(params=[
    {'max': 100, 'lin': 10, 'start': 0,
     'def':
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
    {'max': 99, 'lin': 10, 'start': 0,
     'def':
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]},
    {'max': 87, 'lin': 10, 'start': 0,
     'def':
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 87]},
])
def steps(request):
    """Generate the steps lists."""
    request.param['gen'] = list(generate_steps(request.param['max'],
                                               request.param['lin'],
                                               request.param['start']))
    return request.param


@given(integers(min_value=0))
@settings(deadline=None)
def test_initial_start(initial):
    """Ensure the first value produced is the initial value."""
    gen = generate_steps(total_steps=initial+10, start=initial)
    assert next(gen) == initial


@given(integers(min_value=0))
@settings(deadline=None)
def test_initial_start_series(initial):
    """Ensure the first value produced is the initial value."""
    gen = GenerateStepSeries(total_steps=initial+10)
    assert next(gen) == 0


@pytest.mark.parametrize('final', [100, 10000, 100000, 100001, 99999])
def test_final_total(final):
    """Ensure the final value produced is the final value."""
    genlist = list(generate_steps(total_steps=final))
    assert genlist[-1] == final


@pytest.mark.parametrize('final', [100, 10000, 100000, 100001, 99999])
def test_final_total_series(final):
    """Ensure the final value produced is the final value."""
    genlist = list(GenerateStepSeries(total_steps=final))
    assert genlist[-1] == final


def test_generate_steps(steps):  # pylint: disable=redefined-outer-name
    """Test generation of steps."""
    assert steps['gen'][-1] == steps['max']
    assert steps['gen'] == steps['def']


@pytest.mark.parametrize('total_steps, num_linear', [(10000, 100), (1000000, 100)])
def test_generate_step_series(total_steps, num_linear):
    """Test generate_steps and generate_step_series.

    This ensures that both functions give same results for case of only a
    single series
    """
    single = list(generate_steps(total_steps=total_steps, num_linear=num_linear, start=0))
    series = list(GenerateStepSeries(total_steps=total_steps, num_linear=num_linear, max_gen=1))
    assert single == series


@given(integers(min_value=1, max_value=300))
def test_num_linear(num_linear):
    """Test a range of values of num_linear will work."""
    gen_list = list(generate_steps(total_steps=1e7, num_linear=num_linear))
    assert gen_list[:num_linear+1] == list(range(num_linear+1))


def test_get_index():
    max_gen = 500
    gen_steps = 10
    g = GenerateStepSeries(5000, gen_steps=gen_steps, max_gen=max_gen)
    for _ in g:
        assert len(g.get_index()) <= max_gen
        assert np.all(np.array(g.get_index()) <= max_gen)


def test_generate_step_series_many():
    """Test generation of a step series works."""
    total_steps = 1000000
    num_linear = 10
    many_gens = list(GenerateStepSeries(
        total_steps=total_steps,
        num_linear=num_linear,
        gen_steps=1000,
        max_gen=10,
    ))
    single_gen = list(GenerateStepSeries(
        total_steps=total_steps,
        num_linear=num_linear,
        gen_steps=1000,
        max_gen=1,
    ))
    assert len(many_gens) > len(single_gen)


@pytest.mark.parametrize('total_steps, num_linear', [(1_000_000, 100)])
def test_no_duplicates(total_steps, num_linear):
    """Test generation of a step series works."""
    series_list = list(generate_steps(total_steps=total_steps, num_linear=num_linear))
    series_set = set(series_list)
    assert len(series_list) == len(series_set)


@pytest.mark.parametrize('total_steps, num_linear', [(1_000_000, 100)])
def test_no_duplicates_series(total_steps, num_linear):
    """Test generation of a step series works."""
    series_list = list(GenerateStepSeries(total_steps=total_steps, num_linear=num_linear,
                                          gen_steps=1000, max_gen=10))
    series_set = set(series_list)
    assert len(series_list) == len(series_set)


def test_next():
    """Ensure next() works the same as __iter__."""
    list_iter = list(GenerateStepSeries(total_steps=1000, num_linear=10))
    next_iter = []
    gen = GenerateStepSeries(total_steps=1000, num_linear=10)
    while True:
        try:
            next_iter.append(next(gen))
        except StopIteration:
            break
    assert list_iter == next_iter
