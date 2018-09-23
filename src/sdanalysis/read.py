#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Read input files and compute dynamic and thermodynamic quantities."""

import logging
import multiprocessing
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import attr
import gsd.hoomd
import numpy as np
import pandas

from .dynamics import dynamics, relaxations
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
    while True:
        frame_index = -1
        try:
            return trajectory[frame_index].configuration.step

        # The final configuration is malformed
        except RuntimeError:
            frame_index -= 1


def process_gsd(sim_params: SimulationParams) -> Iterator[Tuple[List[int], HoomdFrame]]:
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
            for frame in src:
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
        curr_step = 0
        for frame in src:
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
                yield step_iter.get_index(), HoomdFrame(frame)

            try:
                curr_step = next(step_iter)
            except StopIteration:
                return


def process_lammpstrj(
    sim_params: SimulationParams
) -> Iterator[Tuple[List[int], LammpsFrame]]:
    indexes = [0]
    assert sim_params.infile is not None
    parser = parse_lammpstrj(sim_params.infile)
    for frame in parser:
        if sim_params.num_steps is not None and frame.timestep > sim_params.num_steps:
            return

        yield indexes, frame


def parse_lammpstrj(filename: Path) -> Iterator[LammpsFrame]:
    logger.debug("Parse file: %s", filename)
    with open(filename) as src:
        while True:
            # Timestep
            line: Union[str, List[str]] = src.readline()
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
    queue: multiprocessing.Queue,
    sim_params: SimulationParams,
    mol_relaxations: List[Dict[str, Any]] = None,
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

    set_filename_vars(sim_params.infile, sim_params)
    if sim_params.outfile is not None:
        dataframes = WriteCache(queue=queue)
    else:
        dataframes = WriteCache(queue=queue)

    keyframes: List[dynamics] = []
    relaxframes: List[relaxations] = []
    if sim_params.infile.suffix == ".gsd":
        file_iterator: FileIterator = process_gsd(sim_params)
    elif sim_params.infile.suffix == ".lammpstrj":
        file_iterator = process_lammpstrj(sim_params)
    for indexes, frame in file_iterator:
        for index in indexes:
            try:
                logger.debug(
                    "len(keyframes): %d, len(relaxframes): %d",
                    len(keyframes),
                    len(relaxframes),
                )
                mydyn = keyframes[index]
                myrelax = relaxframes[index]
            except IndexError:
                logger.debug("Frame: %s", frame)
                logger.debug("Create keyframe at step %s", frame.timestep)
                keyframes.append(
                    dynamics(
                        frame.timestep, frame.box, frame.position, frame.orientation
                    )
                )
                relaxframes.append(
                    relaxations(
                        frame.timestep,
                        frame.box,
                        frame.position,
                        frame.orientation,
                        Trimer(),
                    )
                )
                mydyn = keyframes[index]
                myrelax = relaxframes[index]
                # Set custom relaxation functions
                if mol_relaxations is not None:
                    myrelax.set_mol_relax(mol_relaxations)
            dynamics_series = mydyn.computeAll(
                frame.timestep, frame.position, frame.orientation
            )
            myrelax.add(frame.timestep, frame.position, frame.orientation)
            logger.debug("Series: %s", index)
            dynamics_series["start_index"] = index
            dynamics_series["temperature"] = sim_params.temperature
            dynamics_series["pressure"] = sim_params.pressure
            dataframes.append(dynamics_series)
    if sim_params.outfile is not None:
        dataframes.flush()
        mol_relax = pandas.concat(
            (relax.summary() for relax in relaxframes), keys=range(len(relaxframes))
        )
        mol_relax["temperature"] = sim_params.temperature
        mol_relax["pressure"] = sim_params.pressure
        queue.put(("molecular_relaxations", mol_relax))
        return None

    return dataframes.to_dataframe()
