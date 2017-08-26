# -*- coding: utf-8 -*-
"""Crystals module for generating unit cells for use in hoomd."""

import math

import hoomd
import numpy as np
import quaternion as qt

from . import molecule


class Crystal(object):
    """Defines the base class of a crystal lattice."""

    def __init__(self):
        super().__init__()
        # Hoomd requires that the cell length parameters are a list rather than a tuple.
        self.a1 = [1., 0., 0.]  # pylint: disable=invalid-name
        self.a2 = [0., 1., 0.]  # pylint: disable=invalid-name
        self.a3 = [0., 0., 1.]  # pylint: disable=invalid-name
        self.dimensions = 2
        self._orientations = np.zeros(1)
        self.positions = [[0, 0, 0]]
        self.molecule = molecule.Molecule()

    def get_cell_len(self):
        """Return the unit cell parameters.

        Returns:
            tuple: A tuple containing all the unit cell parameters

        """
        return self.a1, self.a2, self.a3

    def get_abs_positions(self):
        """Return the absolute positions of the molecules.

        This converts the relative positions that the positions are stored in
        with the absolute positions and returns.

        Returns:
            class:`numpy.ndarray`: Positions of each molecule

        """
        return np.dot(np.array(self.positions), self.get_matrix())

    def get_unitcell(self):
        """Return the hoomd unit cell parameter."""
        return hoomd.lattice.unitcell(
            N=self.get_num_molecules(),
            a1=self.a1,
            a2=self.a2,
            a3=self.a3,
            position=self.get_abs_positions(),
            dimensions=self.dimensions,
            orientation=self.get_orientations(),
            type_name=['A'] * self.get_num_molecules(),
            mass=[1.0] * self.get_num_molecules(),
            moment_inertia=([self.molecule.moment_inertia] * self.get_num_molecules())
        )

    def compute_volume(self):
        """Calculate the volume of the unit cell.

        If the number of dimensions is 2, then the returned value will be the
        area rather than the volume.

        Returns:
            float: Volume or area in unitless quantity

        """
        if self.dimensions == 3:
            return np.linalg.norm(np.dot(
                np.array(self.a1),
                np.cross(np.array(self.a2), np.array(self.a3))
            ))
        elif self.dimensions == 2:
            return np.linalg.norm(np.cross(
                np.array(self.a1), np.array(self.a2)))
        else:
            raise ValueError("Dimensions needs to be either 2 or 3")

    def get_matrix(self) -> np.ndarray:
        """Convert the crystal lattice parameters to a matrix.

        Returns:
            class:`np.ndarray`: Matrix of lattice positions

        """
        return np.array([self.a1, self.a2, self.a3])

    def get_orientations(self):
        """Return the orientation quaternions of each molecule.

        Args:
            angle (float): The angle that a molecule is oriented

        Returns:
            class:`numpy.quaternion`: Quaternion representation of the angle

        """
        angles = self._orientations * (math.pi / 180)
        return qt.as_float_array(np.array(
            [qt.from_rotation_vector((0, 0, angle)) for angle in angles]))

    def get_num_molecules(self):
        """Return the number of molecules."""
        return len(self._orientations)


class CrysTrimer(Crystal):
    """A class for the crystal structures of the 2D Trimer molecule."""

    def __init__(self):
        super().__init__()
        self.dimensions = 2
        self.molecule = molecule.Trimer()


class TrimerP2(CrysTrimer):
    """Defining the unit cell of the p2 group of the Trimer molecule."""

    def __init__(self):
        super().__init__()
        self.a1 = [3.82, 0, 0]
        self.a2 = [0.63, 2.55, 0]
        self.a3 = [0, 0, 1]
        self.positions = [[0.3, 0.32, 0], [0.7, 0.68, 0]]
        self._orientations = np.array([50, 230])


class TrimerP2gg(CrysTrimer):
    """Unit cell of p2gg trimer.

    The positions are given in fractional coordinates.

    """

    def __init__(self):
        super().__init__()
        self.a1 = [2.63, 0, 0]
        self.a2 = [0, 7.38, 0]
        self.a3 = [0, 0, 1]
        self.positions = [
            [0.061, 0.853, 0],
            [0.561, 0.647, 0],
            [0.439, 0.353, 0],
            [0.939, 0.147, 0],
        ]
        self._orientations = np.array([24, 156, -24, 204])


class TrimerPg(CrysTrimer):
    """Unit Cell of pg Trimer."""

    def __init__(self):
        super().__init__()
        self.a1 = [2.71, 0, 0]
        self.a2 = [0, 3.63, 0]
        self.a3 = [0, 0, 1]
        self.positions = [
            [0.35, 0.45, 0],
            [0.65, 0.95, 0],
        ]
        self._orientations = np.array([-21, 21])


class CubicSphere(Crystal):
    """Create a simple cubic lattice."""

    def __init__(self):
        super().__init__()
        self.a = 2.
        self.cell_dimensions = 3
        self.molecule = molecule.Sphere()

    def get_unitcell(self):
        return hoomd.lattice.sc(self.a)


class SquareCircle(Crystal):
    """Create a square latttice."""

    def __init__(self):
        super().__init__()
        self.a = 2.
        self.cell_dimensions = 2
        self.molecule = molecule.Disc()

    def get_unitcell(self):
        return hoomd.lattice.sq(self.a)


CRYSTAL_FUNCS = {
    'p2': TrimerP2,
    'p2gg': TrimerP2gg,
    'pg': TrimerPg,
    'CubicSphere': CubicSphere,
    'SquareCircle': SquareCircle,
}
