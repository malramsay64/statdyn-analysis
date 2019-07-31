#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Module to understand gsd files"""

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import gsd.hoomd
from tqdm import tqdm

from ..frame import Frame, HoomdFrame
from ..StepSize import GenerateStepSeries

logger = logging.getLogger(__name__)

gsd_logger = logging.getLogger("gsd")
gsd_logger.setLevel("WARNING")

tqdm_options = {"dynamic_ncols": True}
FileIterator = Iterator[Tuple[List[int], Frame]]


def _get_num_steps(trajectory):
    """Find the number of steps in a Hoomd trajectory handling errors.

    There are cases where it makes sense to run an analysis on the trajectory
    of a simulation which is not yet finished, which has the potential to
    have errors reading the final few frames of the trajectory. This gets the
    number of steps handling the errors in these cases.

    By getting the number of steps in this way, the malformed configurations
    can also be avoided in further processing.

    """
    # Start with the index being the last frame
    frame_index = -1

    # Where there is an error in the last frame, (up to 10)
    # retry with the previous frame.
    max_retries = 10
    for _ in range(max_retries):
        try:
            return trajectory[frame_index].configuration.step
        # The final configuration is malformed, so try and read the previous frame
        except RuntimeError:
            frame_index -= 1

    logger.exception(
        "Read failed. Trajectory length: %s, frame_index: %d",
        len(trajectory),
        frame_index,
    )
    raise RuntimeError("Cannot read frames from trajectory ", trajectory)


def _gsd_linear_trajectory(
    infile: Path,
    steps_max: Optional[int] = None,
    keyframe_interval: int = 20000,
    keyframe_max: int = 500,
):
    # Ensure correct type of infile
    infile = Path(infile)
    index_list: List[int] = []
    with gsd.hoomd.open(str(infile), "rb") as src:
        for index in tqdm(range(len(src)), infile.stem, **tqdm_options):
            try:
                frame = src.read_frame(index)
            except RuntimeError:
                continue
            try:
                timestep = int(frame.configuration.step)
            except IndexError as e:
                logger.error(e)
                raise e
            if timestep % keyframe_interval == 0 and len(index_list) <= keyframe_max:
                index_list.append(len(index_list))
            if steps_max is not None and timestep > steps_max:
                return
            yield index_list, HoomdFrame(frame)
    return


def _gsd_exponential_trajectory(
    infile: Path,
    steps_max: Optional[int] = None,
    keyframe_interval: int = 20000,
    keyframe_max: int = 500,
    linear_steps: int = 100,
):
    with gsd.hoomd.open(str(infile), "rb") as src:
        # Compute steps in gsd file
        if steps_max is None:
            steps_max = _get_num_steps(src)

        # Exponential sequence of steps
        logger.debug("Infile: %s contains %d steps", infile, steps_max)
        step_iter = GenerateStepSeries(
            steps_max,
            num_linear=linear_steps,
            gen_steps=keyframe_interval,
            max_gen=keyframe_max,
        )
        for index in tqdm(range(len(src)), desc=infile.stem, **tqdm_options):
            try:
                frame = src.read_frame(index)
            except RuntimeError:
                logger.info("Found corrupt frame at index %s. Continuing", index)
                continue

            # Increment Step
            try:
                curr_step = int(next(step_iter))
            except StopIteration:
                return

            logger.debug("Step %d with index %s", curr_step, step_iter.get_index())

            # This handles when the generators don't match up
            if not isinstance(curr_step, int):
                raise RuntimeError(
                    f"Expected integer value for current step, got {type(curr_step)}"
                )
            try:
                timestep = int(frame.configuration.step)
            except IndexError as e:
                logger.error(e)
                raise e

            if curr_step > timestep:
                logger.warning(
                    "Step missing in iterator: current %d, frame %d",
                    curr_step,
                    timestep,
                )
                continue

            elif curr_step < timestep:
                logger.warning(
                    "Step missing in gsd trajectory: current %d, frame %d",
                    curr_step,
                    timestep,
                )
                while curr_step < timestep:
                    try:
                        curr_step = next(step_iter)
                    except StopIteration:
                        return

            if steps_max is not None and curr_step > steps_max:
                return

            if curr_step == timestep:
                try:
                    yield step_iter.get_index(), HoomdFrame(frame)
                # Handle error creating a HoomdFrame class
                except ValueError as e:
                    logger.warning(e)
                    continue


def read_gsd_trajectory(
    infile: Path,
    steps_max: Optional[int] = None,
    linear_steps: Optional[int] = 100,
    keyframe_interval: int = 1_000_000,
    keyframe_max: int = 500,
) -> FileIterator:
    """Perform analysis of a GSD file."""
    infile = Path(infile)

    if linear_steps is None:
        yield from _gsd_linear_trajectory(
            infile, steps_max, keyframe_interval, keyframe_max
        )
    else:
        yield from _gsd_exponential_trajectory(
            infile, steps_max, keyframe_interval, keyframe_max, linear_steps
        )
