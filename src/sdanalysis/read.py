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
from typing import Any, Dict, List, Union

import gsd.hoomd
import pandas

from .dynamics import dynamics, relaxations
from .molecules import Trimer
from .params import SimulationParams
from .StepSize import GenerateStepSeries

logger = logging.getLogger(__name__)


def process_gsd(sim_params: SimulationParams):
    with gsd.hoomd.open(sim_params.infile, 'rb') as src:
        # Compute steps in gsd file
        if sim_params.parameters.get('step_limit') is not None:
            num_steps = sim_params.step_limit
        else:
            while True:
                frame_index = -1
                try:
                    num_steps = src[-1].configuration.step
                    break

                except RuntimeError:
                    frame_index -= 1
        logger.debug('Infile: %s contains %d steps', sim_params.infile, num_steps)
        step_iter = GenerateStepSeries(
            num_steps,
            num_linear=sim_params.num_linear,
            gen_steps=sim_params.gen_steps,
            max_gen=sim_params.max_gen,
        )
        curr_step = 0
        for frame in src:
            logger.debug('Step %d with index %s', curr_step, step_iter.get_index())
            # This handles when the generators don't match up
            if curr_step > frame.configuration.step:
                logger.warning(
                    'Step missing in iterator: current %d, frame %d',
                    curr_step,
                    frame.configuration.step,
                )
                continue

            elif curr_step < frame.configuration.step:
                logger.warning(
                    'Step missing in gsd trajectory: current %d, frame %d',
                    curr_step,
                    frame.configuration.step,
                )
                while curr_step < frame.configuration.step:
                    curr_step = step_iter.next()
            if curr_step > num_steps:
                raise StopIteration

            if curr_step == frame.configuration.step:
                yield curr_step, step_iter.get_index(), frame

            curr_step = step_iter.next()


class WriteCache():

    def __init__(
        self, filename: Path, append: bool = True, cache_multiplier: int = 1
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
        pandas.DataFrame.from_records(self._cache).to_hdf(
            self._outfile, 'dynamics', format='table', append=self._append
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


def get_filename_vars(fname: Path):
    fname = Path(fname)
    flist = fname.stem.split('-')
    temp = flist[3][1:]
    pressure = flist[2][1:]
    try:
        crys = flist[4]
    except IndexError:
        crys = None
    variables = namedtuple('variables', ['temperature', 'pressure', 'crystal'])
    return variables(temp, pressure, crys)


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
    if sim_params.infile.endswith('.gsd'):
        file_iterator = process_gsd(sim_params)
    variables = get_filename_vars(Path(sim_params.infile))
    for curr_step, indexes, frame in file_iterator:
        for index in indexes:
            try:
                logger.debug(
                    f'len(keyframes): {len(keyframes)}, len(relaxframes): {len(relaxframes)}'
                )
                mydyn = keyframes[index]
                myrelax = relaxframes[index]
            except IndexError:
                logger.debug('Create keyframe at step %s', curr_step)
                keyframes.append(
                    dynamics(
                        timestep=frame.configuration.step,
                        box=frame.configuration.box,
                        position=frame.particles.position,
                        orientation=frame.particles.orientation,
                    )
                )
                relaxframes.append(
                    relaxations(
                        timestep=frame.configuration.step,
                        box=frame.configuration.box,
                        position=frame.particles.position,
                        orientation=frame.particles.orientation,
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
                curr_step, frame.particles.position, frame.particles.orientation
            )
            myrelax.add(
                curr_step, frame.particles.position, frame.particles.orientation
            )
            logger.debug('Series: %s', index)
            dynamics_series['start_index'] = index
            dynamics_series['temperature'] = variables.temperature
            dynamics_series['pressure'] = variables.pressure
            dataframes.append(dynamics_series)
    if outfile:
        dataframes.flush()
        mol_relax = pandas.concat(
            (relax.summary() for relax in relaxframes), keys=range(len(relaxframes))
        )
        mol_relax['temperature'] = variables.temperature
        mol_relax['pressure'] = variables.pressure
        mol_relax.to_hdf(
            sim_params.outfile, 'molecular_relaxations', format='table', append=True
        )
        return

    return dataframes.to_dataframe()
