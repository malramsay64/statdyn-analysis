#!/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Module to define a molecule to use for simulation."""

import logging
from typing import List, Tuple

import numpy as np

from .math_helper import rotate_vectors

logger = logging.getLogger(__name__)


class Molecule(object):
    """Molecule class holding information on the molecule for use in hoomd.

    This class contains all the paramters required to initialise the molecule
    in a hoomd simulation. This includes all interaction potentials, the rigid
    body interactions and the moments of inertia.

    The Molecule class is a template class that defines a number of functions
    subclasses can use to set these variables however it generates no sensible
    molecule itself.

    """

    def __init__(self) -> None:
        """Initialise defualt properties."""
        self.moment_inertia = (0., 0., 0.)  # type: Tuple[float, float, float]
        self.potential_args = dict()  # type: Dict[Any, Any]
        self.particles = ['A']
        self._radii = {'A': 1.}
        self.dimensions = 3
        self.positions = np.array([[0, 0, 0]])
        self.positions.flags.writeable = False

    def __eq__(self, other) -> bool:
        return type(self) == type(other)

    @property
    def num_particles(self) -> int:
        """Count of particles in the molecule."""
        return len(self.particles)

    def get_types(self) -> List[str]:
        """Get the types of particles present in a molecule."""
        return sorted(list(self._radii.keys()))



    def identify_bodies(self, indexes: np.ndarray) -> np.ndarray:
        """Convert an index of molecules into an index of particles."""
        return np.append(indexes, [indexes]*(self.num_particles-1))

    def __str__(self) -> str:
        return type(self).__name__

    def scale_moment_inertia(self, scale_factor: float) -> None:
        """Scale the moment of inertia by a constant factor."""
        i_x, i_y, i_z = self.moment_inertia
        self.moment_inertia = (i_x * scale_factor, i_y * scale_factor, i_z * scale_factor)

    def compute_moment_intertia(self, scale_factor: float=1) -> Tuple[float, float, float]:
        """Compute the moment of inertia from the particle paramters."""
        positions = self.positions
        COM = np.sum(positions, axis=0)/positions.shape[0]
        moment_inertia = np.sum(np.square(positions - COM))
        moment_inertia *= scale_factor
        return (0, 0, moment_inertia)

    def get_radii(self) -> np.ndarray:
        """Radii of the particles."""
        return np.array([self._radii[p] for p in self.particles])

    def orientation2positions(self, position, orientation):
        return (np.tile(position, (self.num_particles, 1))
                + rotate_vectors(orientation, self.positions.astype(np.float32)))

    def compute_size(self):
        """Compute the maximum possible size of the moleucule.

        This is a rough estimate of the size of the molecule for the creation
        of a lattice that contains no overlaps.

        """
        length = np.max(np.max(self.positions, axis=1) -
                        np.min(self.positions, axis=1))
        return length + 2*self.get_radii().max()


class Disc(Molecule):
    """Defines a 2D particle."""

    def __init__(self) -> None:
        """Initialise 2D disc particle."""
        super().__init__()
        self.dimensions = 2


class Sphere(Molecule):
    """Define a 3D sphere."""

    def __init__(self) -> None:
        """Initialise Spherical particle."""
        super().__init__()


class Trimer(Molecule):
    """Defines a Trimer molecule for initialisation within a hoomd context.

    This defines a molecule of three particles, shaped somewhat like Mickey
    Mouse. The central particle is of type `'A'` while the outer two
    particles are of type `'B'`. The type `'B'` particles, have a variable
    radius and are positioned at a specified distance from the central
    type `'A'` particle. The angle between the two type `'B'` particles,
    subtended by the type `'A'` particle is the other degree of freedom.


    TODO:
        Compute the moment of inertia
    """

    def __init__(self,
                 radius: float=0.637556,
                 distance: float=1.0,
                 angle: float=120,
                 moment_inertia_scale: float=1.) -> None:
        """Initialise trimer molecule.

        Args:
            radius (float): Radius of the small particles. Default is 0.637556
            distance (float): Distance of the outer particles from the central
                one. Default is 1.0
            angle (float): Angle between the two outer particles in degrees.
                Default is 120
            moment_inertia_scale(float): Scale the moment of intertia by this
                factor.

        """
        super().__init__()
        self.radius = radius
        self.distance = distance
        self.angle = angle
        self.particles = ['A', 'B', 'B']
        self._radii.update(B=self.radius)
        self.dimensions = 2
        self.positions = np.array([
            [0, 0, 0],
            [-self.distance * np.sin(self.rad_angle/2), self.distance * np.cos(self.rad_angle/2), 0],
            [self.distance * np.sin(self.rad_angle/2), self.distance * np.cos(self.rad_angle/2), 0],
        ])
        self.positions.flags.writeable = False
        self.moment_inertia = self.compute_moment_intertia(moment_inertia_scale)

    @property
    def rad_angle(self) -> float:
        return np.radians(self.angle)

    def __eq__(self, other) -> bool:
        if super().__eq__(other):
            return (self.radius == other.radius and
                    self.distance == other.distance and
                    self.moment_inertia == other.moment_inertia)
        return False

class Dimer(Molecule):
    """Defines a Dimer molecule for initialisation within a hoomd context.

    This defines a molecule of three particles, shaped somewhat like Mickey
    Mouse. The central particle is of type `'A'` while the outer two
    particles are of type `'B'`. The type `'B'` particles, have a variable
    radius and are positioned at a specified distance from the central
    type `'A'` particle. The angle between the two type `'B'` particles,
    subtended by the type `'A'` particle is the other degree of freedom.

    """

    def __init__(self, radius: float=0.637556, distance: float=1.0) -> None:
        """Intialise Dimer molecule.

        Args:
            radius (float): Radius of the small particles. Default is 0.637556
            distance (float): Distance of the outer particles from the central
                one. Default is 1.0
            angle (float): Angle between the two outer particles in degrees.
                Default is 120

        """
        super(Dimer, self).__init__()
        self.radius = radius
        self.distance = distance
        self.particles = ['A', 'B']
        self._radii.update(B=self.radius)
        self.dimensions = 2
        self.positions = np.array([
            [0, 0, 0],
            [0, self.distance, 0],
        ])
        self.positions.flags.writeable = False
        self.moment_inertia = self.compute_moment_intertia()
