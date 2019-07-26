#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Read input files and compute dynamic and thermodynamic quantities."""

import datetime
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import attr
import gsd.hoomd
import numpy as np
import pandas
from tqdm import tqdm

from .dynamics import Dynamics, Relaxations
from .frame import Frame, HoomdFrame, LammpsFrame
from .molecules import Trimer
from .StepSize import GenerateStepSeries
from .util import get_filename_vars

tqdm_options = {"miniters": 100, "dynamic_ncols": True}

logger = logging.getLogger(__name__)

gsd_logger = logging.getLogger("gsd")
gsd_logger.setLevel("WARNING")


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
    thread_index: int = 0,
):
    # Ensure correct type of infile
    infile = Path(infile)
    index_list: List[int] = []
    with gsd.hoomd.open(str(infile), "rb") as src:
        for index in tqdm(
            range(len(src)), infile.stem, position=thread_index, **tqdm_options
        ):
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
    thread_index: int = 0,
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
        for index in tqdm(
            range(len(src)), desc=infile.stem, position=thread_index, **tqdm_options
        ):
            try:
                frame = src.read_frame(index)
            except RuntimeError:
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


def process_gsd(
    infile: Path,
    steps_max: Optional[int] = None,
    linear_steps: Optional[int] = 100,
    keyframe_interval: int = 1_000_000,
    keyframe_max: int = 500,
    thread_index: int = 0,
) -> FileIterator:
    """Perform analysis of a GSD file."""
    infile = Path(infile)

    if linear_steps is None:
        yield from _gsd_linear_trajectory(
            infile, steps_max, keyframe_interval, keyframe_max, thread_index
        )
    else:
        yield from _gsd_exponential_trajectory(
            infile,
            steps_max,
            keyframe_interval,
            keyframe_max,
            linear_steps,
            thread_index,
        )


def process_lammpstrj(
    infile: Path, steps_max: Optional[int] = None, thread_index: int = 0
) -> Iterator[Tuple[List[int], LammpsFrame]]:

    infile = Path(infile)
    indexes = [0]
    parser = parse_lammpstrj(infile)
    for frame in tqdm(parser, desc=infile.stem, position=thread_index, **tqdm_options):
        if steps_max is not None and frame.timestep > steps_max:
            return

        yield indexes, frame


def parse_lammpstrj(filename: Path) -> Iterator[LammpsFrame]:
    logger.debug("Parse file: %s", filename)
    with open(filename) as src:
        try:
            while True:
                # Timestep
                line: Union[str, List[str]] = src.readline()
                if not line:
                    # We have reached the end of the file
                    return
                assert line == "ITEM: TIMESTEP\n", line
                timestep = int(src.readline().strip())
                logger.debug("Timestep: %d", timestep)
                # Num Atoms
                line = src.readline()
                assert line == "ITEM: NUMBER OF ATOMS\n", line
                num_atoms = int(src.readline().strip())
                logger.debug("num_atoms: %d", num_atoms)
                # Box Bounds
                line = src.readline()
                assert "ITEM: BOX BOUNDS" in line, line
                box_x = src.readline().split()
                box_y = src.readline().split()
                box_z = src.readline().split()
                box = np.array(
                    [
                        float(box_x[1]) - float(box_x[0]),
                        float(box_y[1]) - float(box_y[0]),
                        float(box_z[1]) - float(box_z[0]),
                    ]
                )
                logger.debug("box: %s", box)
                # Atoms
                line = src.readline()
                assert "ITEM: ATOMS" in line, line
                headings = line.strip().split(" ")[2:]
                logger.debug("headings: %s", headings)
                # Find id column
                id_col = headings.index("id")
                # Create arrays
                frame = {
                    field: np.empty(num_atoms, dtype=np.float32) for field in headings
                }
                logger.debug('Array shape of "id": %s', frame["id"].shape)
                for _ in range(num_atoms):
                    line = src.readline().split(" ")
                    mol_index = int(line[id_col]) - 1  # lammps 1 indexes molecules
                    for field, val in zip(headings, line):
                        frame[field][mol_index] = float(val)
                frame["box"] = box
                frame["timestep"] = timestep
                yield LammpsFrame(frame)
        except AssertionError as e:
            raise RuntimeError("Unable to parse trajectory. Found error", e)


@attr.s(auto_attribs=True)
class WriteCache:
    _filename: Optional[Path] = None
    group: str = "dynamics"
    cache_multiplier: int = 1
    to_append: bool = False
    queue: Optional[multiprocessing.Queue] = None

    _cache: List[Any] = attr.ib(default=attr.Factory(list), init=False)
    _cache_default: int = attr.ib(default=8192, init=False)
    _emptied_count: int = attr.ib(default=0, init=False)

    def __attr_post_init__(self):
        if self.filename and self.queue:
            raise ValueError(
                "Can only output to a single source, either filename or queue"
            )
        if self.group is None:
            raise ValueError("Group can not be None.")

    @property
    def _cache_size(self) -> int:
        return self.cache_multiplier * self._cache_default

    def append(self, item: Any) -> None:
        # Cache of size 0 or with val None will act as list
        if self._cache and len(self._cache) == self._cache_size:
            self.flush()
            self._emptied_count += 1
        self._cache.append(item)

    def _flush_queue(self, df) -> None:
        assert self.queue is not None
        assert df is not None
        self.queue.put((self.group, df))

    def _flush_file(self, df) -> None:
        assert self.filename is not None
        assert self.group is not None
        df.to_hdf(self.filename, self.group, format="table", append=self.to_append)
        self.to_append = True

    def flush(self) -> None:
        df = self.to_dataframe()
        if self.queue:
            self._flush_queue(df)
        else:
            self._flush_file(df)
        self._cache.clear()

    @property
    def filename(self) -> Optional[Path]:
        if self._filename is not None:
            return Path(self._filename)
        return None

    def __len__(self) -> int:
        # Total number of elements added
        return self._cache_size * self._emptied_count + len(self._cache)

    def to_dataframe(self):
        return pandas.DataFrame.from_records(self._cache)


def process_file(
    infile: Path,
    wave_number: float,
    steps_max: Optional[int] = None,
    linear_steps: Optional[int] = None,
    keyframe_interval: int = 1_000_000,
    keyframe_max: int = 500,
    mol_relaxations: List[Dict[str, Any]] = None,
    outfile: Optional[Path] = None,
    queue: Optional[multiprocessing.Queue] = None,
    thread_index: int = 0,
) -> Optional[pandas.DataFrame]:
    """Read a file and compute the dynamics quantities.

    This computes the dynamic quantities from a file returning the
    result as a pandas DataFrame. This is only suitable for cases where
    all the data will fit in memory, as there is no writing to a file.

    Args:

    Returns:
        (py:class:`pandas.DataFrame`): DataFrame with the dynamics quantities.

    """
    assert infile is not None
    infile = Path(infile)

    logger.info("Processing %s", infile)
    start_time = datetime.datetime.now()

    sim_variables = get_filename_vars(infile)
    if outfile is not None and queue is None:
        dataframes = WriteCache(filename=outfile)
    elif queue:
        dataframes = WriteCache(queue=queue)
    else:
        dataframes = WriteCache()

    keyframes: Dict[int, Tuple[Dynamics, Relaxations]] = {}
    if infile.suffix == ".gsd":
        file_iterator: FileIterator = process_gsd(
            infile=infile,
            steps_max=steps_max,
            linear_steps=linear_steps,
            keyframe_interval=keyframe_interval,
            keyframe_max=keyframe_max,
            thread_index=thread_index,
        )
    elif infile.suffix == ".lammpstrj":
        file_iterator = process_lammpstrj(
            infile, steps_max=steps_max, thread_index=thread_index
        )
    for indexes, frame in file_iterator:
        if frame.position.shape[0] == 0:
            logger.warning("Found malformed frame in %s... continuing", infile.name)
            continue

        for index in indexes:
            dyn, relax = keyframes.setdefault(
                index,
                (
                    Dynamics.from_frame(frame, Trimer(), wave_number),
                    Relaxations.from_frame(frame, Trimer(), wave_number),
                ),
            )
            # Set custom relaxation functions
            if mol_relaxations is not None:
                relax.set_mol_relax(mol_relaxations)

            try:
                dynamics_series = dyn.compute_all(
                    frame.timestep, frame.position, frame.orientation, frame.image
                )
                relax.add(frame.timestep, frame.position, frame.orientation)
            except (ValueError, RuntimeError) as e:
                logger.warning(e)
                continue

            logger.debug("Series: %s", index)
            dynamics_series["start_index"] = index
            dynamics_series["temperature"] = sim_variables.temperature
            dynamics_series["pressure"] = sim_variables.pressure
            dataframes.append(dynamics_series)

    end_time = datetime.datetime.now()
    processing_time = end_time - start_time

    logger.info("Finished processing %s, took %s", infile, processing_time)

    if outfile is not None:
        dataframes.flush()
        mol_relax = pandas.concat(
            (r.summary() for _, r in keyframes.values()), keys=list(keyframes.keys())
        )
        mol_relax["temperature"] = sim_variables.temperature
        mol_relax["pressure"] = sim_variables.pressure
        if queue:
            queue.put(("molecular_relaxations", mol_relax))
        else:
            mol_relax.to_hdf(outfile, "molecular_relaxations")
        return None

    return dataframes.to_dataframe()


def open_trajectory(filename: Path):
    filename = Path(filename)
    if filename.suffix == ".gsd":
        with gsd.hoomd.open(str(filename)) as trj:
            for frame in trj:
                yield HoomdFrame(frame)
    elif filename.suffix == ".lammpstrj":
        trj = parse_lammpstrj(filename)
        for frame in trj:
            yield frame
