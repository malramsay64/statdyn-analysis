#!/usr/bin/env python
""" A series of classes that specify various step size functions"""

from numba import jit


@jit
def generate_steps(total_steps, num_linear=99, start=0):
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

        >>> [s for s in generate_steps(100, 9)]
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    """
    curr_step = start
    step_size = 1
    lin_steps = 0
    curr_step += step_size
    while curr_step < total_steps:
        if lin_steps == num_linear:
            step_size *= 10
            lin_steps = 0
        yield curr_step
        lin_steps += 1
        curr_step += step_size
    yield total_steps


def generate_step_series(total_steps, num_linear=99, gen_steps=300000, max_gen=500, index=False):
    gen = generate_steps(total_steps, num_linear, 0)
    curr_step = next(gen)
    generators = [(next(gen), gen)]
    argmin = 0
    try:
        while curr_step <= total_steps:
            if index:
                yield curr_step, argmin
            else:
                yield curr_step
            if curr_step % gen_steps == 0 and curr_step > 0 and len(generators) < max_gen:
                gen = generate_steps(total_steps, num_linear, curr_step)
                generators.append((curr_step, gen))
            argmin = min(enumerate(generators), key=lambda x: x[1][0])[0]
            curr_step, gen = generators[argmin]
            generators[argmin] = (next(gen), gen)
    except StopIteration:
        if index:
            yield curr_step, argmin
        else:
            yield curr_step

