#!/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Module to define a molecule to use for simulation."""

import logging
from itertools import combinations_with_replacement
from typing import Any, Dict, List, Tuple

import hoomd
import hoomd.md
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
        self.potential = hoomd.md.pair.lj
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

    def define_dimensions(self) -> None:
        """Set the number of dimensions for the simulation.

        This takes into accout the number of dimensions of the molecule,
        a 2D molecule can only be a 2D molecule, since there will be no
        rotations in that 3rd dimension anyway.
        """
        if self.dimensions == 2:
            hoomd.md.update.enforce2d()

    def define_potential(self) -> hoomd.md.pair.pair:
        r"""Define the potential in the simulation context.

        A helper function that defines the potential to be used by the  hoomd
        simulation context. The default values for the potential are a
        Lennard-Jones potential with a cutoff of :math:`2.5\sigma` and
        interaction parameters of :math:`\epsilon = 1.0` and
        :math:`\sigma = 2.0`.

        Returns:
            class:`hoomd.md.pair`: The interaction potential object class.

        """
        self.potential_args.setdefault('r_cut', 2.5)
        potential = self.potential(
            **self.potential_args,
            nlist=hoomd.md.nlist.cell()
        )
        for i, j in combinations_with_replacement(self._radii.keys(), 2):
            potential.pair_coeff.set(i, j, epsilon=1, sigma=self._radii[i] + self._radii[j])
        return potential

    def define_rigid(self, params: Dict[Any, Any]=None
                     ) -> hoomd.md.constrain.rigid:
        """Define the rigid constraints of the molecule.

        This is a helper function to define the rigid body constraints of the
        particular molecules within the hoomd context.

        Args:
            create (bool): Flag that toggles the option of creating the
                additional particles when creating the rigid bodies. Defaults
                to False.
            params (dict): Dictionary defining the rigid body structure. The
                default values for the `type_name` of A and the `types` of the
                `self.particles` variable should work for the vast majority of
                systems, so the only value required should be the topology.

        Returns:
            class:`hoomd.md.constrain.rigid`: Rigid constraint object

        """
        if len(self.particles) <= 1:
            logger.info("Not enough particles for a rigid body")
            return
        if not params:
            params = dict()
        params['type_name'] = self.particles[0]
        params['types'] = self.particles[1:]
        params.setdefault('positions', [tuple(pos) for i, pos in enumerate(self.positions) if i > 0])
        rigid = hoomd.md.constrain.rigid()
        rigid.set_param(**params)
        return rigid

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
        return (np.repeat(position, self.num_particles, axis=0)
                + rotate_vectors(orientation, self.positions.astype(np.float32)))

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
