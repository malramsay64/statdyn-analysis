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
from freud import Box

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

    if len(flist) < 4:
        return variables(None, None, None)

    pressure_str = flist[2]
    if pressure_str[0] == "P":
        pressure: Optional[str] = pressure_str[1:]
    else:
        pressure = None

    temperature_str = flist[3]
    if temperature_str[0] == "T":
        temp: Optional[str] = temperature_str[1:]
    else:
        temp = None

    if len(flist) >= 5:
        crys: Optional[str] = flist[4]
    else:
        crys = None

    return variables(temp, pressure, crys)


def set_filename_vars(fname: PathLike, sim_params: SimulationParams) -> None:
    """Set the variables of the simulations params according to the filename."""
    var = get_filename_vars(fname)
    for attr in ["temperature", "pressure", "crystal"]:
        if getattr(var, attr) is not None:
            value = getattr(var, attr)
            if attr not in ["crystal"]:
                value = float(value)
            setattr(sim_params, attr, value)


def quaternion_rotation(initial, final, result):
    result[:] = rowan.geometry.intrinsic_distance(initial, final)


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


def displacement_periodic(box, initial, final, result):
    if len(box) > 3 and np.any(box[3:] != 0.):
        raise NotImplementedError(
            "Periodic distances for non-orthorhombic boxes are not yet implemented."
            f"Got xy: {box[3]}, xz: {box[4]}, yz: {box[5]}"
        )
    delta = np.abs(final - initial)
    result[:] = np.linalg.norm(
        np.where(delta > 0.5 * box[:3], delta - box[:3], delta), axis=1
    )


def orientation2positions(
    mol: Molecule, position: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    return np.tile(position, (mol.num_particles, 1)) + np.concatenate(
        [rotate_vectors(orientation, pos) for pos in mol.positions.astype(np.float32)]
    )
