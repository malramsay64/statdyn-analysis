#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A series of classes that specify various step size functions."""

from typing import Iterable, Iterator, List  # pylint: disable=unused-import


def generate_steps(total_steps: int,
                   num_linear: int=100,
                   start: int=0) -> Iterator[int]:
    """Generate a sequence of steps with a power law.

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


class GenerateStepSeries(Iterable):
    """Generate a many sequences of steps with different starting values."""

    def __init__(self,
                 total_steps: int,
                 num_linear: int=100,
                 gen_steps: int=300000,
                 max_gen: int=500) -> None:
        self.total_steps = total_steps
        self.num_linear = num_linear
        self.gen_steps = gen_steps
        self.max_gen = max_gen
        gen = generate_steps(self.total_steps, self.num_linear, 0)
        self.generators = [(next(gen), gen)]
        self.argmin = 0
        self.stop_iteration = False

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self.stop_iteration:
            raise StopIteration
        self.argmin = min(enumerate(self.generators),
                          key=lambda x: x[1][0])[0]
        curr_step, gen = self.generators[self.argmin]
        try:
            self.generators[self.argmin] = (next(gen), gen)
            if (self.gen_steps > 0
                    and curr_step % self.gen_steps == 0
                    and len(self.generators) < self.max_gen):
                self.generators.append(
                    (curr_step,
                     generate_steps(self.total_steps,
                                    self.num_linear,
                                    curr_step))
                )
            return curr_step
        except StopIteration:
            self.stop_iteration = True
            return curr_step
        return self.total_steps

    def get_index(self) -> int:
        """Return the index from which the previous step was returned."""
        return self.argmin

    def next(self) -> int:
        """Return the next value of the iterator."""
        return self.__next__()
