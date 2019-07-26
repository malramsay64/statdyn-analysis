#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""A collection of utility functions."""

import logging
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Union

import numpy as np
import rowan
from freud.box import Box

from .molecules import Molecule
from .params import SimulationParams

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class Variables(NamedTuple):
    temperature: Optional[str]
    pressure: Optional[str]
    crystal: Optional[str]
    iteration_id: Optional[str]

    @classmethod
    def from_filename(cls, fname: PathLike) -> "Variables":
        """Create a Variables instance taking information from a path.

        This extracts the information about the value of variables used within a simulation
        trajectory from the filename. This is expecting the information in a specific
        format, where values are separated by the dash character `-`.

        Args:
            fname: The full path of the filename from which to extract the information.

        .. warn::

            This is expecting the full name of the file, including the extension. Should
            there not be an extension on the filename, values could be stripped giving
            undefined behaviour.

        """
        fname = Path(fname)
        flist = fname.stem.split("-")
        logger.debug("Split Filename: %s", str(flist))

        if flist[0] in ["dump", "trajectory", "thermo"]:
            del flist[0]

        # The remaining three quantities being molecule, temperature and pressure
        if len(flist) < 3:
            return Variables(None, None, None, None)

        pressure: Optional[str] = None
        temperature: Optional[str] = None
        iteration_id: Optional[str] = None
        crystal: Optional[str] = None

        for item in flist:
            if item[0] == "P":
                pressure = item[1:]
            elif item[0] == "T":
                temperature = item[1:]
            elif item[:2] == "ID":
                iteration_id = item[2:]
            else:
                crystal = item

        return cls(temperature, pressure, crystal, iteration_id)


def get_filename_vars(fname: PathLike) -> Variables:
    """Extract variables information from a filename.

    This extracts the information about the value of variables used within a simulation
    trajectory from the filename. This is expecting the information in a specific
    format, where values are separated by the dash character `-`.

    Args:
        fname: The full path of the filename from which to extract the information.

    .. warn::

        This is expecting the full name of the file, including the extension. Should
        there not be an extension on the filename, values could be stripped giving
        undefined behaviour.

    """

    return Variables.from_filename(fname)


def set_filename_vars(fname: PathLike, sim_params: SimulationParams) -> None:
    """Set the variables of the simulations params according to the filename."""
    var = get_filename_vars(fname)
    for attr in ["temperature", "pressure"]:
        if getattr(var, attr) is not None:
            value = float(getattr(var, attr))
            setattr(sim_params, attr, value)

    if var.crystal is not None:
        sim_params.space_group = var.crystal


def parse_directory(directory: Path, glob: str) -> Dict[str, Dict]:
    directory = Path(directory)
    all_values: Dict[str, Dict] = {}
    files = list(directory.glob(glob))
    logger.debug("Found files: %s", files)
    for fname in files:
        temperature, pressure, crystal, repr_index = get_filename_vars(fname)
        assert temperature
        assert pressure
        crystal = str(crystal)
        repr_index = str(repr_index)
        all_values.setdefault(pressure, {})
        all_values[pressure].setdefault(temperature, {})
        all_values[pressure][temperature].setdefault(crystal, {})
        all_values[pressure][temperature][crystal].setdefault(repr_index, {})
        all_values[pressure][temperature][crystal][repr_index] = fname
    return all_values


def quaternion_rotation(initial, final):
    return rowan.geometry.intrinsic_distance(initial, final)


def rotate_vectors(quaternion, vector):
    return rowan.rotate(quaternion, vector)


def quaternion_angle(quaternion) -> np.ndarray:
    return rowan.geometry.angle(quaternion)


def z2quaternion(theta: np.ndarray) -> np.ndarray:
    """Convert a rotation about the z axis to a quaternion.

    This is a helper for 2D simulations, taking the rotation of a particle about the z axis and
    converting it to a quaternion. The input angle `theta` is assumed to be in radians.

    """
    return rowan.from_euler(theta, 0, 0).astype(np.float32)


def quaternion2z(quaternion: np.ndarray) -> np.ndarray:
    """Convert a rotation about the z axis to a quaternion.

    This is a helper for 2D simulations, taking the rotation of a particle about the z axis and
    converting it to a quaternion. The input angle `theta` is assumed to be in radians.

    """
    return rowan.to_euler(quaternion)[:, 0].astype(np.float32)


def orientation2positions(
    mol: Molecule, position: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    return np.tile(position, (mol.num_particles, 1)) + np.concatenate(
        [rotate_vectors(orientation, pos) for pos in mol.positions]
    )


def create_freud_box(box: np.ndarray, is_2D=True) -> Box:
    """Convert an array of box values to a box for use with freud functions

    The freud package has a special type for the description of the simulation cell, the
    Box class. This is a function to take an array of lengths and tilts to simplify the
    creation of the Box class for use with freud.

    """
    # pylint: disable=invalid-name
    Lx, Ly, Lz = box[:3]
    xy = xz = yz = 0
    if len(box) == 6:
        xy, xz, yz = box[3:6]
    if is_2D:
        return Box(Lx=Lx, Ly=Ly, xy=xy, is2D=is_2D)
    return Box(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz)
    # pylint: enable=invalid-name
