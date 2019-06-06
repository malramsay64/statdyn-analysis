#!/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Module to define a molecule to use for simulation."""

import logging
from typing import Dict, List

import attr
import numpy as np

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True, cmp=False)
class Molecule:
    """A template class for the generation of molecules for analysis.

    The positions of all molecules will be adjusted to ensure the center of mass is at
    the position (0, 0, 0).

    """

    dimensions: int = 3
    particles: List[str] = attr.ib(default=attr.Factory(lambda: ["A"]))
    positions: np.ndarray = attr.ib(
        default=attr.Factory(lambda: np.zeros((1, 3))), repr=False, cmp=False
    )
    _radii: Dict[str, float] = attr.ib(default=attr.Factory(dict))
    rigid: bool = False
    moment_inertia_scale: float = 1

    def __attrs_post_init__(self) -> None:
        self._radii.setdefault("A", 1.0)
        self.positions -= self._compute_center_of_mass()

    @property
    def num_particles(self) -> int:
        """ Count of particles in the molecule"""
        return len(self.particles)

    def _compute_center_of_mass(self) -> np.ndarray:
        return np.mean(self.positions, axis=0)

    def get_types(self) -> List[str]:
        """Get the types of particles present in a molecule."""
        return sorted(list(self._radii.keys()))

    def __str__(self) -> str:
        return type(self).__name__

    def get_radii(self) -> np.ndarray:
        """Radii of the particles."""
        return np.array([self._radii[p] for p in self.particles])


class Disc(Molecule):
    """Defines a 2D particle."""

    def __init__(self) -> None:
        """Initialise 2D disc particle."""
        super().__init__(dimensions=2)


class Sphere(Molecule):
    """Define a 3D sphere."""

    def __init__(self) -> None:  # pylint: disable=useless-super-delegation
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


    """

    def __init__(
        self,
        radius: float = 0.637_556,
        distance: float = 1.0,
        angle: float = 120,
        moment_inertia_scale: float = 1.0,
    ) -> None:
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
        if not isinstance(radius, (float, int)):
            raise ValueError(
                f"The parameter 'radius' needs to be specified as a float, got, {type(radius)}"
            )
        if not isinstance(distance, (float, int)):
            raise ValueError(
                f"The parameter 'distance' needs to be specified as a float, got, {type(distance)}"
            )
        if not isinstance(angle, (float, int)):
            raise ValueError(
                f"The parameter 'angle' needs to be specified as a float, got, {type(angle)}"
            )

        super().__init__()
        self.radius = radius
        self.distance = distance
        self.angle = angle
        rad_ang = np.deg2rad(angle)
        particles = ["A", "B", "B"]
        radii = {"A": 1.0, "B": self.radius}
        positions = np.array(
            [
                [0, 0, 0],
                [-distance * np.sin(rad_ang / 2), distance * np.cos(rad_ang / 2), 0],
                [distance * np.sin(rad_ang / 2), distance * np.cos(rad_ang / 2), 0],
            ]
        )
        super().__init__(
            positions=positions,
            dimensions=2,
            radii=radii,
            particles=particles,
            rigid=True,
            moment_inertia_scale=moment_inertia_scale,
        )

    @property
    def rad_angle(self) -> float:
        return np.radians(self.angle)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(radius={self.radius}, distance={self.distance}, "
            f"angle={self.angle}, moment_inertia_scale={self.moment_inertia_scale})"
        )

    def __eq__(self, other) -> bool:
        if super().__eq__(other):
            return (
                self.radius == other.radius
                and self.distance == other.distance
                and self.angle == other.angle
            )

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

    def __init__(
        self,
        radius: float = 0.637_556,
        distance: float = 1.0,
        moment_inertia_scale: float = 1,
    ) -> None:
        """Initialise Dimer molecule.

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
        particles = ["A", "B"]
        radii = {"A": 1.0, "B": self.radius}
        positions = np.array([[0, 0, 0], [0, self.distance, 0]])
        super().__init__(
            dimensions=2,
            particles=particles,
            positions=positions,
            radii=radii,
            rigid=True,
            moment_inertia_scale=moment_inertia_scale,
        )

    def __eq__(self, other) -> bool:
        if super().__eq__(other):
            return self.radius == other.radius and self.distance == other.distance

        return False
