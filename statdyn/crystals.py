"""Crystals module for generating unit cells for use in hoomd"""

import math
import numpy as np
import hoomd
import quaternion as qt
from statdyn import molecule


class Crystal(object):
    """Defines the base class of a crystal lattice"""
    def __init__(self):
        super().__init__()
        self.a1 = [1, 0, 0]
        self.a2 = [0, 1, 0]
        self.a3 = [0, 0, 1]
        self.dimensions = 2
        self._orientations = np.zeros(1)
        self.positions = [[0, 0, 0]]
        self.molecule = molecule.Molecule()


    def get_cell_len(self):
        """Return the unit cell parameters
        Returns:
            tuple: A tuple containing all the unit cell parameters

        """
        return self.a1, self.a2, self.a3

    def get_unitcell(self):
        """Return the hoomd unit cell parameter"""
        return hoomd.lattice.unitcell(
            N=self.get_num_molecules(),
            a1=self.a1,
            a2=self.a2,
            a3=self.a3,
            position=self.positions,
            dimensions=self.dimensions,
            orientation=self.get_orientations(),
            type_name=['A']*self.get_num_molecules(),
            mass=[1.0]*self.get_num_molecules(),
            moment_inertia=[self.molecule.moment_inertia]*self.get_num_molecules()
        )

    def compute_volume(self):
        if self.dimensions == 3:
            return np.linalg.norm(np.dot(
                np.array(self.a1),
                np.cross(np.array(self.a2), np.array(self.a3))
            ))
        elif self.dimensions == 2:
            return np.linalg.norm(np.cross(
                np.array(self.a1),
                np.array(self.a2)
            ))
        else:
            raise ValueError("Dimensions needs to be either 2 or 3")

    def get_matrix(self):
        return np.array([self.a1, self.a2, self.a3])

    def get_orientations(self):
        """Return the orientation quaternions of each molecule

        Args:
            angle (float): The angle that a molecule is oriented

        Returns:
            class:`numpy.quaternion`: Quaternion representation of the angle
        """
        angles = self._orientations*(math.pi/180)
        return qt.as_float_array(np.array(
            [qt.from_rotation_vector((0, 0, angle)) for angle in angles]))

    def get_num_molecules(self):
        """Return the number of molecules"""
        return len(self._orientations)


class CrysTrimer(Crystal):
    """A class for the crystal structures of the 2D Trimer molecule"""
    def __init__(self):
        super().__init__()
        self.dimensions = 2
        self.molecule = molecule.Trimer()


class p2(CrysTrimer):
    """Defining the unit cell of the p2 group of the Trimer molecule"""
    def __init__(self):
        super().__init__()
        self.a1 = [3.82, 0, 0]
        self.a2 = [0.68, 2.53, 0]
        self.a3 = [0, 0, 1]
        self.positions = [[0.3, 0.32, 0], [0.7, 0.68, 0]]
        self._orientations = np.array([40, -140])
