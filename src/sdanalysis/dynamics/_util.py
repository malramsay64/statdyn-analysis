#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Helper functions for the dynamics module."""

from functools import lru_cache
from typing import Optional

import freud.density
import numpy as np
import rowan
from freud.box import Box

from ..util import rotate_vectors


# This decorator is what enables the caching of this function,
# making this function 100 times faster for subsequent executions
@lru_cache()
def create_wave_vector(wave_number: float, angular_resolution: int):
    r"""Convert a wave number into a radially symmetric wave vector.

    This calculates the values of cos and sin :math:`\theta` for `angular_resolution`
    values of :math:`\theta` between 0 and :math:`2\pi`.

    The results of this function are cached, so these values only need to be computed
    a single time, the rest of the time they are just returned.

    """
    angles = np.linspace(0, 2 * np.pi, num=angular_resolution, endpoint=False).reshape(
        (-1, 1)
    )
    wave_vector = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
    return wave_vector * wave_number


def molecule2particles(
    position: np.ndarray, orientation: np.ndarray, mol_vector: np.ndarray
) -> np.ndarray:
    if np.allclose(orientation, 0.0):
        orientation[:, 0] = 1.0
    return np.concatenate(
        [rotate_vectors(orientation, pos) for pos in mol_vector.astype(np.float32)]
    ) + np.repeat(position, mol_vector.shape[0], axis=0)


def _static_structure_factor(
    rdf: freud.density.RDF, wave_number: float, num_particles: int
):
    dr = rdf.R[1] - rdf.R[0]
    integral = dr * np.sum((rdf.RDF - 1) * rdf.R * np.sin(wave_number * rdf.R))
    density = num_particles / rdf.box.volume
    return 1 + 4 * np.pi * density / wave_number * integral


def calculate_wave_number(box: Box, positions: np.ndarray):
    """Calculate the wave number for a configuration.

    It is not recommended to automatically compute the wave number, since this will
    lead to potentially unusual results.

    """
    rmax = min(box.Lx / 2.2, box.Ly / 2.2)
    if not box.is2D:
        rmax = min(rmax, box.Lz / 2.2)

    dr = rmax / 200
    rdf = freud.density.RDF(dr=dr, rmax=rmax)
    rdf.compute(box, positions)

    ssf = []
    x = np.linspace(0.5, 20, 200)
    for value in x:
        ssf.append(_static_structure_factor(rdf, value, len(positions)))

    return x[np.argmax(ssf)]


class TrackedMotion:
    """Keep track of the motion of a particle allowing for multiple periods.

    This keeps track of the position of a particle as each frame is added, which allows
    for tracking the motion of a particle through multiple periods, as long as each
    motion takes the shortest distance.

    """

    box: Box

    # Keeping track of the total overall motion
    delta_translation: np.ndarray
    delta_rotation: np.ndarray

    # Keeping track of the previous position
    previous_position: np.ndarray
    previous_orientation: Optional[np.ndarray]

    def __init__(
        self, box: Box, position: np.ndarray, orientation: Optional[np.ndarray]
    ):
        """

        Args:
            box: The dimensions of the simulation cell, allowing the calculation
                of periodic distance.
            position: The position of each particle, given as an array with shape
                (N, 3) where N is the number of particles.
            orientation: The orientation of each particle, which is represented
                as a quaternion.

        """
        self.box = box
        self.previous_position = position
        self.delta_translation = np.zeros_like(position)
        self.delta_rotation = np.zeros_like(position)
        if orientation is not None:
            if orientation.shape[0] == 0:
                raise RuntimeError("Orientation must contain values, has length of 0.")
            self.previous_orientation = orientation
        else:
            self.previous_orientation = None

    def add(self, position: np.ndarray, orientation: Optional[np.ndarray]):
        """Update the state of the dynamics calculations by adding the next values.

        This updates the motion of the particles, comparing the positions and
        orientations of the current frame with the previous frame, adding the difference
        to the total displacement. This approach allows for tracking particles over
        periodic boundaries, or through larger rotations assuming that there are
        sufficient frames to capture the information. Each single displacement obeys the
        minimum image convention, so for large time intervals it is still possible to
        have missing information.

        Args:
            position: The current positions of the particles
            orientation: The current orientations of the particles represented as a quaternion

        """
        self.delta_translation -= self.box.wrap(self.previous_position - position)
        if self.previous_orientation is not None and orientation is not None:
            self.delta_rotation += rowan.to_euler(
                rowan.divide(orientation, self.previous_orientation)
            )

        self.previous_position = position
        self.previous_orientation = orientation
