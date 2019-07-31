#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A series of classes that specify various step size functions."""

import logging
from collections import namedtuple
from itertools import takewhile
from queue import Empty, PriorityQueue
from typing import Dict, Iterator, List

logger = logging.getLogger(__name__)


iterindex = namedtuple("iterindex", ["index", "iterator"])
stepiterindex = namedtuple("stepiterindex", ["step", "index", "iterator"])


def generate_steps(
    total_steps: int, num_linear: int = 100, start: int = 0
) -> Iterator[int]:
    """Generate a sequence of steps with a power law.

    This is a function for generating a sequence of steps such that they create
    a continuous curve when plotted on a log scale. The idea is that at long
    timescales, data only needs to be collected very infrequently compared to
    short timescales.  By changing the rate at which we collect the data as the
    timescale increases it drastically reduces the amount of data required for
    capture, storage, processing, and visualisation.

    Args:
        total_steps: The total number of steps required for the
            simulation, i.e. the value you want to stop on.
        num_linear: The number of linear steps before increasing the size
            of the steps by a power of 10. There is always an extra step in the
            first sequence, this is to make the patterns nicer.
            The default value of 99 gives dense data across the plot.
        start: The starting value (default is 0) if the data capture is
            commenced at a timestep after (or before) 0.

    Example:

        >>> [s for s in generate_steps(100, 10)]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    """
    yield from takewhile(
        lambda x: x < total_steps,
        exp_sequence(start, num_linear, initial_step_size=1, base=10),
    )
    yield total_steps


def exp_sequence(
    start: int = 0, num_linear: int = 100, initial_step_size: int = 1, base: int = 10
) -> Iterator[int]:
    """Sequence of integers in an exponential like sequence.

    This function generates integers in a sequence consisting of two components,
    a linear sequence, which is interspersed with an exponential increase in
    the size of each linear step. This sequence is designed to be useful for
    selecting points to plot on a logarithmic scale.

    The default values represent a good mix between fewer points and having good
    coverage of the scale on a logarithmic plot.

    Note that the first value returned by this function will always be the
    `start` argument.

    Args:
        start (int): The starting value of the sequence. This only shifts the
            sequence, the difference between points is the same regardless of the
            start value. (default = 0)
        num_linear (int): The number of linear steps before increasing the size
            of each linear step. (default = 100)
        initial_step_size (int): The size of the first set of linear steps.
            (default = 1)
        base (int): The base of the exponent of which the initial_step_size is
            increased. (default = 10)
    """
    step_size = initial_step_size
    curr_step = start
    yield curr_step

    while True:
        for step in (start + (i) * step_size for i in range(num_linear + 1)):
            if step > curr_step:
                yield step

        curr_step = start + num_linear * step_size
        logger.debug("Current step %d", curr_step)
        step_size *= base


class GenerateStepSeries:
    """Generate a many sequences of steps with different starting values."""

    def __init__(
        self,
        total_steps: int,
        num_linear: int = 100,
        gen_steps: int = 200000,
        max_gen: int = 500,
    ) -> None:
        """"""
        self.total_steps = total_steps
        self.num_linear = num_linear
        self.gen_steps = gen_steps
        self.max_gen = max_gen
        self.curr_step = 0
        self._num_generators = 0
        self.values: Dict[int, List[iterindex]] = {}
        self._queue: PriorityQueue = PriorityQueue()
        self._add_generator()

    def _enqueue(self, iindex: iterindex) -> None:
        try:
            step = next(iindex.iterator)
        except StopIteration:
            return

        if step in self.values:
            temp = self.values.get(step)
            assert temp is not None
            temp.append(iindex)
        else:
            self.values[step] = [iindex]
            self._queue.put(step)

    def _add_generator(self) -> None:
        new_gen = generate_steps(self.total_steps, self.num_linear, self.curr_step)
        self._enqueue(iterindex(self._num_generators, new_gen))
        logger.debug("Generator added with index %d", self._num_generators)
        self._num_generators += 1

    def __iter__(self):
        return self

    def __next__(self) -> int:
        # Dequeue
        previous_step = self.curr_step
        try:
            self.curr_step = self._queue.get(block=False)
        except Empty:
            raise StopIteration

        # Cleanup from previous step
        if self.curr_step != previous_step:
            del self.values[previous_step]
        elif previous_step != 0:
            pass
        # Check for new indexes
        if self.curr_step % self.gen_steps == 0 and self.curr_step > 0:
            if self._num_generators < self.max_gen:
                self._add_generator()
        # Get list of indexes
        iterindexes = self.values.get(self.curr_step)
        logger.debug("Value of iterindexes: %s at step %d", iterindexes, self.curr_step)
        assert iterindexes is not None
        # Add interators back onto queue
        for iindex in iterindexes:
            self._enqueue(iindex)
        # Return
        return self.curr_step

    def get_index(self) -> List[int]:
        """Return the indexes from which the previous step was returned."""
        return [i.index for i in self.values.get(self.curr_step, [])]

    def next(self) -> int:
        """Return the next value of the iterator."""
        return self.__next__()
