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
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import attr
import gsd.hoomd
import numpy as np
import pandas

from ..dynamics import Dynamics, Relaxations
from ..frame import Frame, HoomdFrame
from ..molecules import Trimer
from ..util import get_filename_vars
from ._gsd import FileIterator, read_gsd_trajectory
from ._lammps import parse_lammpstrj, read_lammps_trajectory

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class WriteCache:
    _filename: Optional[Path] = None
    group: str = "dynamics"
    cache_multiplier: int = 1
    to_append: bool = False

    _cache: List[Any] = attr.ib(default=attr.Factory(list), init=False)
    _cache_default: int = attr.ib(default=8192, init=False)
    _emptied_count: int = attr.ib(default=0, init=False)

    def __attr_post_init__(self):
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

    def _flush_file(self, df) -> None:
        assert self.filename is not None
        assert self.group is not None
        df.to_hdf(self.filename, self.group, format="table", append=self.to_append)
        self.to_append = True

    def flush(self) -> None:
        df = self.to_dataframe()
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
    if outfile is not None:
        dataframes = WriteCache(filename=outfile)
    else:
        dataframes = WriteCache()

    keyframes: Dict[int, Tuple[Dynamics, Relaxations]] = {}
    if infile.suffix == ".gsd":
        file_iterator: FileIterator = read_gsd_trajectory(
            infile=infile,
            steps_max=steps_max,
            linear_steps=linear_steps,
            keyframe_interval=keyframe_interval,
            keyframe_max=keyframe_max,
        )
    elif infile.suffix == ".lammpstrj":
        file_iterator = read_lammps_trajectory(infile, steps_max=steps_max)
    for indexes, frame in file_iterator:
        if frame.position.shape[0] == 0:
            logger.warning("Found malformed frame in %s... continuing", infile.name)
            continue

        logger.info("Indexes for step %s: %s", frame.timestep, indexes)
        for index in indexes[0:1]:
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
            dynamics_series["keyframe"] = index
            if sim_variables.temperature is None:
                dynamics_series["temperature"] = np.nan
            else:
                dynamics_series["temperature"] = float(sim_variables.temperature)
            if sim_variables.pressure is None:
                dynamics_series["pressure"] = np.nan
            else:
                dynamics_series["pressure"] = float(sim_variables.pressure)
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
        mol_relax.index.names = ["keyframe", "molecule"]
        mol_relax = mol_relax.reset_index()
        mol_relax.to_hdf(outfile, "molecular_relaxations")
        return None

    return dataframes.to_dataframe()


def open_trajectory(filename: Path) -> Iterator[Frame]:
    filename = Path(filename)
    if filename.suffix == ".gsd":
        with gsd.hoomd.open(str(filename)) as trj:
            for index in range(len(trj)):
                try:
                    yield HoomdFrame(trj[index])
                except RuntimeError:
                    logger.info("Found corrupt frame at index %s continuing", index)
                    continue
    elif filename.suffix == ".lammpstrj":
        trj = parse_lammpstrj(filename)
        for frame in trj:
            yield frame
