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
from .params import SimulationParams
from .StepSize import GenerateStepSeries
from .util import set_filename_vars

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


def process_gsd(sim_params: SimulationParams, thread_index: int = 0) -> FileIterator:
    """Perform analysis of a GSD file.

    This is a specialisation of the process_file, called when the extension of a file is
    `.gsd`; as such it shouldn't typically be called by the user. For the user facing
    function see :func:`process_file`.

    """
    assert sim_params.infile is not None
    assert sim_params.gen_steps is not None
    assert sim_params.max_gen is not None
    with gsd.hoomd.open(str(sim_params.infile), "rb") as src:

        # Compute steps in gsd file
        if sim_params.num_steps is not None:
            num_steps = sim_params.num_steps
        else:
            num_steps = _get_num_steps(src)

        # Return the steps in sequence. This allows a linear sequence of steps.
        if sim_params.linear_steps is None:
            index_list = []
            for frame in tqdm(src, position=thread_index, miniters=100):
                if (
                    frame.configuration.step % sim_params.gen_steps == 0
                    and len(index_list) <= sim_params.max_gen
                ):
                    index_list.append(len(index_list))
                if frame.configuration.step > num_steps:
                    return
                yield index_list, HoomdFrame(frame)
            return

        # Exponential sequence of steps
        logger.debug("Infile: %s contains %d steps", sim_params.infile, num_steps)
        step_iter = GenerateStepSeries(
            num_steps,
            num_linear=sim_params.linear_steps,
            gen_steps=sim_params.gen_steps,
            max_gen=sim_params.max_gen,
        )
        for frame in tqdm(
            src, desc=sim_params.infile.stem, position=thread_index, miniters=100
        ):
            # Increment Step
            try:
                curr_step = next(step_iter)
            except StopIteration:
                return
            logger.debug("Step %d with index %s", curr_step, step_iter.get_index())
            # This handles when the generators don't match up
            if curr_step > frame.configuration.step:
                logger.warning(
                    "Step missing in iterator: current %d, frame %d",
                    curr_step,
                    frame.configuration.step,
                )
                continue

            elif curr_step < frame.configuration.step:
                logger.warning(
                    "Step missing in gsd trajectory: current %d, frame %d",
                    curr_step,
                    frame.configuration.step,
                )
                while curr_step < frame.configuration.step:
                    try:
                        curr_step = next(step_iter)
                    except StopIteration:
                        return
            if curr_step > num_steps:
                return

            if curr_step == frame.configuration.step:
                try:
                    yield step_iter.get_index(), HoomdFrame(frame)
                # Handle error creating a HoomdFrame class
                except ValueError as e:
                    logger.warning(e)
                    continue


def process_lammpstrj(
    sim_params: SimulationParams, thread_index: int = 0
) -> Iterator[Tuple[List[int], LammpsFrame]]:
    indexes = [0]
    assert sim_params.infile is not None
    parser = parse_lammpstrj(sim_params.infile)
    for frame in tqdm(parser, position=thread_index, miniters=100):
        if sim_params.num_steps is not None and frame.timestep > sim_params.num_steps:
            return

        yield indexes, frame


def parse_lammpstrj(filename: Path) -> Iterator[LammpsFrame]:
    logger.debug("Parse file: %s", filename)
    with open(filename) as src:
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
            frame = {field: np.empty(num_atoms, dtype=np.float32) for field in headings}
            logger.debug('Array shape of "id": %s', frame["id"].shape)
            for _ in range(num_atoms):
                line = src.readline().split(" ")
                mol_index = int(line[id_col]) - 1  # lammps 1 indexes molecules
                for field, val in zip(headings, line):
                    frame[field][mol_index] = float(val)
            frame["box"] = box
            frame["timestep"] = timestep
            yield LammpsFrame(frame)


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
    sim_params: SimulationParams,
    mol_relaxations: List[Dict[str, Any]] = None,
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
    assert sim_params.infile is not None
    logger.info("Processing %s", sim_params.infile)
    start_time = datetime.datetime.now()

    set_filename_vars(sim_params.infile, sim_params)
    if sim_params.outfile is not None and queue is None:
        dataframes = WriteCache(filename=sim_params.outfile)
    elif queue:
        dataframes = WriteCache(queue=queue)
    else:
        dataframes = WriteCache()

    keyframes: Dict[int, Tuple[Dynamics, Relaxations]] = {}
    if sim_params.infile.suffix == ".gsd":
        file_iterator: FileIterator = process_gsd(sim_params, thread_index=thread_index)
    elif sim_params.infile.suffix == ".lammpstrj":
        file_iterator = process_lammpstrj(sim_params, thread_index=thread_index)
    for indexes, frame in file_iterator:
        if frame.position.shape[0] == 0:
            logger.warning(
                "Found malformed frame in %s... continuing", sim_params.infile.name
            )
            continue

        for index in indexes:
            mydyn, myrelax = keyframes.setdefault(
                index,
                (
                    Dynamics.from_frame(frame, Trimer()),
                    Relaxations.from_frame(frame, Trimer()),
                ),
            )
            # Set custom relaxation functions
            if mol_relaxations is not None:
                myrelax.set_mol_relax(mol_relaxations)

            try:
                dynamics_series = mydyn.compute_all(
                    frame.timestep, frame.position, frame.orientation, frame.image
                )
                myrelax.add(frame.timestep, frame.position, frame.orientation)
            except (ValueError, RuntimeError) as e:
                logger.warning(e)
                continue

            logger.debug("Series: %s", index)
            dynamics_series["start_index"] = index
            dynamics_series["temperature"] = sim_params.temperature
            dynamics_series["pressure"] = sim_params.pressure
            dataframes.append(dynamics_series)

    end_time = datetime.datetime.now()
    processing_time = end_time - start_time

    logger.info("Finished processing %s, took %s", sim_params.infile, processing_time)

    if sim_params.outfile is not None:
        dataframes.flush()
        mol_relax = pandas.concat(
            (relax.summary() for _, relax in keyframes.values()),
            keys=list(keyframes.keys()),
        )
        mol_relax["temperature"] = sim_params.temperature
        mol_relax["pressure"] = sim_params.pressure
        if queue:
            queue.put(("molecular_relaxations", mol_relax))
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
