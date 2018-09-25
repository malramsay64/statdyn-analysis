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
from copy import deepcopy
from multiprocessing import Manager, Pool, Queue, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas

from .params import SimulationParams
from .read import process_file


def file_writer(queue: Queue, outfile: Path):
    with pandas.HDFStore(outfile, "w") as dst:
        while True:
            group, dataset = queue.get()
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
) -> None:
    # The manager queue needs to be used
    manager = Manager()
    queue = manager.Queue()

    # Number of cpus + an additional for the writer file
    with Pool(cpu_count() + 1) as pool:

        # Put file writing process to work first
        writer = pool.apply_async(file_writer, (queue, sim_params.outfile))

        # Fire off worker processes
        # starmap allows for passing multiple args to process_file
        pool.starmap(
            process_file,
            ((queue, _set_input_file(sim_params, i), relaxations) for i in input_files),
        )

        # Send None to file writer to kill
        queue.put(None)
        # Wait for the file writer to finish writing
        writer.wait()
