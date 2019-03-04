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
from typing import NamedTuple, Optional, Union

import numpy as np
import rowan

from .molecules import Molecule
from .params import SimulationParams

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class variables(NamedTuple):
    temperature: Optional[str]
    pressure: Optional[str]
    crystal: Optional[str]


def get_filename_vars(fname: PathLike):
    fname = Path(fname)
    flist = fname.stem.split("-")
    logger.debug("Split Filename: %s", str(flist))

    if flist[0] in ["dump", "trajectory", "thermo"]:
        del flist[0]

    # The remaining three quantities being molecule, temperature and pressure
    if len(flist) < 3:
        return variables(None, None, None)

    pressure: Optional[str] = None
    temperature: Optional[str] = None

    for item in flist:
        if item[0] == "P":
            pressure = item[1:]
        elif item[0] == "T":
            temperature = item[1:]

    # The 4th item has to be the crystal structure
    if len(flist) >= 4:
        crys: Optional[str] = flist[3]
    else:
        crys = None

    return variables(temperature, pressure, crys)


def set_filename_vars(fname: PathLike, sim_params: SimulationParams) -> None:
    """Set the variables of the simulations params according to the filename."""
    var = get_filename_vars(fname)
    for attr in ["temperature", "pressure"]:
        if getattr(var, attr) is not None:
            value = float(getattr(var, attr))
            setattr(sim_params, attr, value)

    if var.crystal is not None:
        sim_params.space_group = var.crystal


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
