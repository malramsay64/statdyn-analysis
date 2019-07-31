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
from gsd.hoomd import HOOMDTrajectory
from tqdm import tqdm

from ..frame import Frame, HoomdFrame
from ..StepSize import GenerateStepSeries

logger = logging.getLogger(__name__)

gsd_logger = logging.getLogger("gsd")
gsd_logger.setLevel("WARNING")

tqdm_options = {"dynamic_ncols": True}
FileIterator = Iterator[Tuple[List[int], Frame]]


def iter_trajectory(trajectory: HOOMDTrajectory) -> Iterator[HoomdFrame]:
    """Create an iterator over a HOOMDTrajectory object.

    This iterates over the snapshots in a HOOMDTrajectory from reading a gsd file. This
    is an alternative to the inbuilt iterator of the HOOMDTrajectory object which
    performs error handling, which is primarily discarding corrupted frames.

    Args:
        trajectory: The output of py:func:`gsd.hoomd.open` which is to be iterated over.

    Returns:
        An iterator over the trajectory returning HoomdFrame objects.

    This is primarily intended as an internal helper function, performing the error
    handling in a single location which makes it easier to deal with.

    """
    for index in range(len(trajectory)):
        try:
            yield HoomdFrame(trajectory[index])
        except RuntimeError:
            logger.info("Found corrupt frame at index %s. Continuing", index)


def _get_num_steps(trajectory: gsd.hoomd.HOOMDTrajectory) -> int:
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
        except IndexError as e:
            logger.error(e)
            raise e

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
        for frame in tqdm(iter_trajectory(src), infile.stem, **tqdm_options):
            # Update keyframe index
            if (
                frame.timestep % keyframe_interval == 0
                and len(index_list) <= keyframe_max
            ):
                index_list.append(len(index_list))

            # Stopping condition
            if steps_max is not None and frame.timestep > steps_max:
                return

            yield index_list, frame
    return


def _gsd_exponential_trajectory(
    infile: Path,
    steps_max: Optional[int] = None,
    keyframe_interval: int = 20000,
    keyframe_max: int = 500,
    linear_steps: int = 100,
):
    # Start by logging errors to warning
    log_level = logger.warning
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
        traj_iter = iter_trajectory(src)
        for frame in tqdm(traj_iter, desc=infile.stem, **tqdm_options):
            # Increment Step
            try:
                curr_step = int(next(step_iter))
                assert isinstance(curr_step, int)
            except StopIteration:
                return

            logger.debug("Step %d with index %s", curr_step, step_iter.get_index())

            # There are steps in the trajectory which don't appear in the iterator
            # Catch up the trajectory to align with the iterator.
            if curr_step > frame.timestep:
                log_level(
                    "Step missing in iterator: iterator %d, trajectory %d",
                    curr_step,
                    frame.timestep,
                )
                # Subsequent errors will be logged to debug
                log_level = logger.debug
                while frame.timestep < curr_step:
                    try:
                        frame = next(traj_iter)
                    except StopIteration:
                        return

            # There are steps in the iterator which don't appear in the trajectory
            # Catch up the iterator to align with the trajectory.
            elif curr_step < frame.timestep:
                log_level(
                    "Step missing in gsd trajectory: iterator %d, trajectory %d",
                    curr_step,
                    frame.timestep,
                )
                # Subsequent errors will be logged to debug
                log_level = logger.debug
                while curr_step < frame.timestep:
                    try:
                        curr_step = next(step_iter)
                    except StopIteration:
                        return

            # Stopping condition -> When we have gone beyond the number of steps we want
            if steps_max is not None and curr_step > steps_max:
                return

            if curr_step == frame.timestep:
                yield step_iter.get_index(), frame


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
