#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Read input files and compute dynamic and thermodynamic quantities."""

import logging
from typing import List

import gsd.hoomd
import pandas

from ..StepSize import GenerateStepSeries
from .dynamics import dynamics

logger = logging.getLogger(__name__)


def process_gsd(infile: str,
                gen_steps: int=20000,
                max_gen: int=1000,
                num_linear: int=100,
                step_limit: int=None,
                outfile: str=None,
                buffer_multiplier: int=1,
                ) -> pandas.DataFrame:
    """Read a gsd file and compute the dynamics quantities.

    This computes the dynamic quantities from a gsd file returning the
    result as a pandas DataFrame. This is only suitable for cases where
    all the data will fit in memory, as there is no writing to a file.

    Args:
        infile (str): The filename of the gsd file from which to read the
            configurations.
        gen_steps (int): The value of the parameter `gen_steps` used when
            running the dynamics simulation. (default: 20000)
        step_limit (int): Limit the timescale of the processing. A value of
            ``None`` (default) will process all steps in the file.
        outfile (str): When present write the results to a file rather than
            returning from the function. The hdf5 file is written throughout the
            analysis allowing for results that are too large to completely
            fit in memory. The write process is buffered to improve performance.
        buffer_multiplier (int): When writing a file this is a multiplier for
            the number of dataframes to buffer before writing. This should be
            tailored to the specific memory requirements of the machine being
            used. A multiplier of 1 (default) uses about 150 MB of RAM.

    Returns:
        (py:class:`pandas.DataFrame`): DataFrame with the dynamics quantities.

    """
    dataframes = []  # type: List[pandas.DataFrame]
    keyframes = []  # type: List[dynamics]
    append_file = False
    buffer_size = int(buffer_multiplier * 8192)

    curr_step = 0
    with gsd.hoomd.open(infile, 'rb') as src:
        if step_limit:
            num_steps = step_limit
        else:
            num_steps = src[-1].configuration.step
        logger.debug('Infile: %s contains %d steps', infile, num_steps)
        step_iter = GenerateStepSeries(num_steps,
                                       num_linear=num_linear,
                                       gen_steps=gen_steps,
                                       max_gen=max_gen)
        for frame in src:
            logger.debug('Step %d with index %s',
                         curr_step, step_iter.get_index())

            # This handles when the generators don't match up
            if curr_step > frame.configuration.step:
                logger.warning('Step missing in iterator: current %d, frame %d',
                               curr_step, frame.configuration.step)
                continue

            elif curr_step < frame.configuration.step:
                logger.warning('Step missing in gsd trajectory: current %d, frame %d',
                               curr_step, frame.configuration.step)
                try:
                    while curr_step < frame.configuration.step:
                        curr_step = step_iter.next()
                except StopIteration:
                    break

            if curr_step == frame.configuration.step:
                indexes = step_iter.get_index()
                for index in indexes:
                    try:
                        mydyn = keyframes[index]
                    except IndexError:
                        logger.debug('Create keyframe at step %s', curr_step)
                        keyframes.append(dynamics(
                            timestep=frame.configuration.step,
                            box=frame.configuration.box,
                            position=frame.particles.position,
                            orientation=frame.particles.orientation,
                        ))
                        mydyn = keyframes[index]

                    dynamics_series = mydyn.computeAll(
                        curr_step,
                        frame.particles.position,
                        frame.particles.orientation
                    )
                    logger.debug('Series: %s', index)
                    dynamics_series['start_index'] = index
                    dataframes.append(dynamics_series)

                try:
                    curr_step = step_iter.next()
                except StopIteration:
                    break

                if outfile and len(dataframes) >= buffer_size:
                    pandas.concat(dataframes).to_hdf(
                        outfile,
                        'dynamics',
                        format='table',
                        append=append_file,
                    )
                    dataframes.clear()
                    # Once we have written to the file once, append to the
                    # existing file.
                    if not append_file:
                        append_file = True

    if outfile:
        pandas.concat(dataframes).to_hdf(
            outfile,
            'dynamics',
            format='table',
            append=append_file,
        )
        return

    return pandas.concat(dataframes)
