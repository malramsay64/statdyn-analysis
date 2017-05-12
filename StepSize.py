#!/usr/bin/env python
""" A series of classes that specify various step size functions"""

from numba import jit


@jit
def generate_steps(total_steps, num_linear=99, start=0):
    """Generate a sequence of steps with a power law

    A sequence of steps which consisting of points spaced by a
    power law which contain a number of linear steps between each jump.

    Args:
        num_linear (int): The number of linear steps between each jump in power.
            The distance between these steps is going to increase as the gap
            between powers become larger. In the case when there num_linear is
            greater than the number of steps between powers every step is
            returned.
        pow_jump (int): The jumps in powers of 10

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
