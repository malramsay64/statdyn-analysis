#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Read input files and compute dynamic and thermodynamic quantities."""

import logging
from collections import namedtuple
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import gsd.hoomd
import numpy as np
import pandas

from .dynamics import dynamics, relaxations
from .frame import Frame, gsdFrame, lammpsFrame
from .molecules import Trimer
from .params import SimulationParams
from .StepSize import GenerateStepSeries
from .util import get_filename_vars

logger = logging.getLogger(__name__)


def process_gsd(sim_params: SimulationParams) -> Iterable[Tuple[List[int], gsdFrame]]:
    with gsd.hoomd.open(sim_params.infile, "rb") as src:
        # Compute steps in gsd file
        if sim_params.parameters.get("step_limit") is not None:
            num_steps = sim_params.step_limit
        else:
            while True:
                frame_index = -1
                try:
                    num_steps = src[-1].configuration.step
                    break

                except RuntimeError:
                    frame_index -= 1
        logger.debug("Infile: %s contains %d steps", sim_params.infile, num_steps)
        step_iter = GenerateStepSeries(
            num_steps,
            num_linear=sim_params.num_linear,
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
                    curr_step = step_iter.next()
            if curr_step > num_steps:
                return

            if curr_step == frame.configuration.step:
                yield step_iter.get_index(), gsdFrame(frame)

            curr_step = step_iter.next()


def process_lammpstrj(
    sim_params: SimulationParams
) -> Iterable[Tuple[List[int], lammpsFrame]]:
    indexes = [0]
    parser = parse_lammpstrj(sim_params.infile)
    frame = next(parser)
    while frame:
        if frame.timestep > sim_params.step_limit:
            return

        yield indexes, frame

        frame = next(parser)


def parse_lammpstrj(filename: Path, mode: str = "r") -> Iterable[lammpsFrame]:
    logger.debug("Parse file: %s", filename)
    with open(filename) as src:
        while True:
            # Timestep
            line = src.readline()
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
            yield lammpsFrame(frame)


class WriteCache:
    def __init__(
        self, filename: Path, cache_multiplier: int = 1, append: bool = False
    ) -> None:
        self._cache_size = 8192 * cache_multiplier
        self._cache = []  # type: List[Any]
        self._outfile = filename
        self._append = append
        self._emptied = 0

    def append(self, item: Any) -> None:
        # Cache of size 0 or with val None will act as list
        if self._cache and len(self._cache) == self._cache_size:
            self.flush()
            self._emptied += 1
        self._cache.append(item)

    def flush(self) -> None:
        self.to_dataframe().to_hdf(
            self._outfile, "dynamics", format="table", append=self._append
        )
        self._append = True
        self._cache.clear()

    def get_outfile(self) -> Path:
        return Path(self._outfile)

    def __len__(self) -> int:
        # Total number of elements added
        return self._cache_size * self._emptied + len(self._cache)

    def to_dataframe(self):
        return pandas.DataFrame.from_records(self._cache)


def process_file(sim_params: SimulationParams) -> None:
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
    try:
        outfile = Path(sim_params.outfile)
        dataframes = WriteCache(outfile, append=True)
    except AttributeError:
        outfile = None
        dataframes = WriteCache(outfile, append=True, cache_multiplier=0)
    keyframes: List[dynamics] = []
    relaxframes: List[relaxations] = []
    if sim_params.infile.endswith(".gsd"):
        file_iterator = process_gsd(sim_params)
    elif sim_params.infile.endswith(".lammpstrj"):
        file_iterator = process_lammpstrj(sim_params)
    variables = get_filename_vars(sim_params.infile)
    for indexes, frame in file_iterator:
        for index in indexes:
            try:
                logger.debug(
                    f"len(keyframes): {len(keyframes)}, len(relaxframes): {len(relaxframes)}"
                )
                mydyn = keyframes[index]
                myrelax = relaxframes[index]
            except IndexError:
                logger.debug("Frame: %s", frame)
                logger.debug("Create keyframe at step %s", frame.timestep)
                keyframes.append(
                    dynamics(
                        timestep=frame.timestep,
                        box=frame.box,
                        position=frame.position,
                        orientation=frame.orientation,
                    )
                )
                relaxframes.append(
                    relaxations(
                        timestep=frame.timestep,
                        box=frame.box,
                        position=frame.position,
                        orientation=frame.orientation,
                        molecule=Trimer(),
                    )
                )
                mydyn = keyframes[index]
                myrelax = relaxframes[index]
                try:
                    myrelax.set_mol_relax(sim_params.mol_relaxations)
                except (KeyError, AttributeError):
                    pass
            dynamics_series = mydyn.computeAll(
                frame.timestep, frame.position, frame.orientation
            )
            myrelax.add(frame.timestep, frame.position, frame.orientation)
            logger.debug("Series: %s", index)
            dynamics_series["start_index"] = index
            dynamics_series["temperature"] = variables.temperature
            dynamics_series["pressure"] = variables.pressure
            dataframes.append(dynamics_series)
    if outfile:
        dataframes.flush()
        mol_relax = pandas.concat(
            (relax.summary() for relax in relaxframes), keys=range(len(relaxframes))
        )
        mol_relax["temperature"] = variables.temperature
        mol_relax["pressure"] = variables.pressure
        mol_relax.to_hdf(
            sim_params.outfile, "molecular_relaxations", format="table", append=True
        )
        return

    return dataframes.to_dataframe()
