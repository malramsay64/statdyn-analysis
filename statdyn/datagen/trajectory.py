#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Series of functions for manipulating data from MD trajectories."""

from pathlib import Path

import gsd.hoomd

from ..StepSize import GenerateStepSeries
from ..TimeDep import TimeDepMany


def compute_motion(filename: Path):
    """Compute the translations and rotations of every molecule."""
    with gsd.hoomd.open(str(filename), 'rb') as src:
        num_steps = src[-1].configuration.step
        step_iter = GenerateStepSeries(num_steps,
                                       num_linear=100,
                                       gen_steps=50000,
                                       max_gen=1000)
        dynamics = TimeDepMany()
        curr_step = 0
        for frame in src:
            dynamics.append(frame,
                            step_iter.get_index(), curr_step)
            curr_step = step_iter.next()
            while curr_step == frame.configuration.step:
                dynamics.append(frame,
                                step_iter.get_index(), curr_step)
                curr_step = step_iter.next()
    return dynamics.get_datafile()
