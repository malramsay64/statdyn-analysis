#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Read and parse Lammps files

This understands the lammpstrj file format.

"""

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from ..frame import LammpsFrame
from ._gsd import tqdm_options

logger = logging.getLogger(__name__)


def read_lammps_trajectory(
    infile: Path, steps_max: Optional[int] = None
) -> Iterator[Tuple[List[int], LammpsFrame]]:

    infile = Path(infile)
    indexes = [0]
    parser = parse_lammpstrj(infile)
    for frame in tqdm(parser, desc=infile.stem, **tqdm_options):
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
