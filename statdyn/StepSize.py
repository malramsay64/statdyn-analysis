#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A series of classes that specify various step size functions"""

from typing import Iterator

from numba import jit


@jit
def generate_steps(total_steps: int,
                   num_linear: int=100,
                   start: int=0) -> Iterator[int]:
    """Generate a sequence of steps with a power law

    This is a function for generating a sequence of steps such that they create
    a continous curve when plotted on a log scale. The idea is that at long
    timescales, data only needs to be collected very infrequently compared to
    short timescales.  By changing the rate at which we collect the data as the
    timescale increases it drastically reduces the amount of data required for
    capture, storage, processing, and visualisation.

    Args:
        total_steps (int): The total number of steps required for the
            simulation, i.e. the value you want to stop on.
        num_linear (int): The numer of linear steps before increasing the size
            of the steps by a power of 10. There is always an extra step in the
            first sequence, this is to make the patterns nicer.
            The default value of 99 gives dense data across the plot.
        start int): The starting value (default is 0) if the data capture is
            commenced at a timestep after (or before) 0.

    Example:

        >>> [s for s in generate_steps(100, 10)]
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    """
    curr_step = start
    step_size = 1
    curr_step += step_size
    while curr_step < total_steps:
        if curr_step - start == step_size * num_linear:
            step_size *= 10
        yield curr_step
        curr_step += step_size
    yield total_steps


def generate_step_series(total_steps: int,
                         num_linear: int=100,
                         gen_steps: int=300000,
                         max_gen: int=500,
                         ret_index: bool=False):
    """Generate a many sequences of steps with different starting values
    """
    gen = generate_steps(total_steps, num_linear, 0)
    curr_step = next(gen)
    generators = [(next(gen), gen)]
    argmin = 0
    try:
        while curr_step <= total_steps:
            if ret_index:
                yield curr_step, argmin
            else:
                yield curr_step
            if (curr_step % gen_steps == 0
                    and curr_step > 0
                    and len(generators) < max_gen):
                gen = generate_steps(total_steps, num_linear, curr_step)
                generators.append((curr_step, gen))
            argmin = min(enumerate(generators), key=lambda x: x[1][0])[0]
            curr_step, gen = generators[argmin]
            generators[argmin] = (next(gen), gen)
    except StopIteration:
        if ret_index:
            yield curr_step, argmin
        else:
            yield curr_step
