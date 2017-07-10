#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Series of functions for manipulating data from MD trajectories."""

from pathlib import Path
import logging

import gsd.hoomd

from ..StepSize import GenerateStepSeries
from ..TimeDep import TimeDepMany


logger = logging.getLogger('motion')
logger.setLevel(logging.DEBUG)


def compute_motion(filename: Path, outdir: Path=None):
    """Compute the translations and rotations of every molecule."""
    with gsd.hoomd.open(str(filename), 'rb') as src:
        num_steps = src[-1].configuration.step
        step_iter = GenerateStepSeries(num_steps+1,
                                       num_linear=10,
                                       gen_steps=20000,
                                       max_gen=1000)
        if outdir:
            outfile = outdir / filename.name
        else:
            outfile = filename
        outfile.with_suffix('.hdf5')
        dynamics = TimeDepMany(outfile)
        curr_step = 0
        for frame in src:
            if curr_step == frame.configuration.step:
                dynamics.append(frame,
                                step_iter.get_index(), curr_step)
                curr_step = step_iter.next()
            elif curr_step > frame.configuration.step:
                logger.warning(
                    'Frame step index missing for current frame, skipping' +
                    ' current step %s, frame step %s, step index %s',
                    curr_step, frame.configuration.step, step_iter.get_index())
            elif curr_step < frame.configuration.step:
                logger.warning(
                    'Current step is behind for current frame, incrementing'
                    ' current step %s, frame step %s, step index %s',
                    curr_step, frame.configuration.step, step_iter.get_index())
                curr_step = step_iter.next()
            # Deal with case where there are multiple generators with the same
            # step in the sequence.
            repeat_counter = 0
            log_step = 0
            while curr_step == frame.configuration.step:
                dynamics.append(frame,
                                step_iter.get_index(), curr_step)
                repeat_counter += 1
                log_step = curr_step
                curr_step = step_iter.next()
            if repeat_counter:
                logger.info('Repeating step %s; %s repetitions',
                            log_step, repeat_counter)
    return dynamics.get_datafile()
