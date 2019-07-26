#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Module to manage the multiprocessing capabilities of this package.

This is segregated to make it simpler to remove, test and work on without affecting
the rest of the implementation.

"""
import functools
import logging
import os
from copy import deepcopy
from multiprocessing import Manager, Pool, Queue, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import freud
import pandas
from tqdm._tqdm import TqdmDefaultWriteLock

from .params import SimulationParams
from .read import process_file

TqdmDefaultWriteLock.create_mp_lock()

logger = logging.getLogger(__name__)


def file_writer(queue: Queue, outfile: Path):
    with pandas.HDFStore(outfile, "w") as dst:
        while True:
            group, dataset = queue.get()
            logger.debug("Writing dataset to %s", dataset)
            if dataset is None:
                break
            dst.append(group, dataset)
        dst.flush()


def _set_input_file(sim_params: SimulationParams, infile: str) -> SimulationParams:
    with sim_params.temp_context(infile=infile):
        return deepcopy(sim_params)


def parallel_process_files(
    input_files: Tuple[str],
    sim_params: SimulationParams,
    relaxations: List[Dict[str, Any]] = None,
    num_cpus: Optional[int] = None,
) -> None:
    # The manager queue needs to be used
    manager = Manager()
    queue = manager.Queue()

    # Only use a single thread for freud
    freud.parallel.setNumThreads(1)
    # Only use a single thread for numpy
    try:
        import mkl

        mkl.set_num_threads(1)
    except ImportError:
        pass
    # Set 1 thread for openmp and openblas
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Number of cpus + an additional for the writer file
    if num_cpus is None:
        num_cpus = cpu_count() + 1
    # When there are fewer than 2 threads the processing never starts
    if num_cpus < 2:
        num_cpus = 2

    par_process = functools.partial(
        process_file,
        wave_number=sim_params.wave_number,
        steps_max=sim_params.num_steps,
        linear_steps=sim_params.linear_steps,
        gen_steps=sim_params.gen_steps,
        max_gen=sim_params.max_gen,
        mol_relaxations=relaxations,
        outfile=None,
        queue=queue,
    )

    with Pool(num_cpus) as pool:

        # Put file writing process to work first
        writer = pool.apply_async(file_writer, (queue, sim_params.outfile))

        # Fire off worker processes
        # starmap allows for passing multiple args to process_file
        pool.starmap(
            par_process,
            (
                {"infile": infile, "thread_index": index}
                for index, infile in enumerate(input_files)
            ),
        )

        # Send None to file writer to kill
        queue.put(None)
        # Wait for the file writer to finish writing
        writer.wait()
