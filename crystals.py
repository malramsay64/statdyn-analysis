"""Crystals module for generating unit cells for use in hoomd"""

import math
import numpy as np
import hoomd
import quaternion
import molecule


class Crystal(object):
    """Defines the base class of a crystal lattice"""
    def __init__(self):
        super().__init__()
        self._a1 = [1, 0, 0]
        self._a2 = [0, 1, 0]
        self._a3 = [0, 0, 1]
        self._dimensions = [2]
        self._orientations = np.ndarray([0])
        self._positions = [[0, 0, 0]]

    def get_dimensions(self):
        """Get the number of dimensions in a lattice
        Returns:
            int: Number of dimensions

        """
        return self._dimensions

    def get_positions(self):
        """Convert fractional coordinates to coordinates for hoomd"""
        tmp_positions = []
        matrix = self.get_matrix()
        for pos in self._positions:
            tmp_positions.append(np.dot(pos, matrix) - self.get_matrix()*0.5)
        return tmp_positions

    def get_cell_len(self):
        """Return the unit cell parameters
        Returns:
            tuple: A tuple containing all the unit cell parameters

        """
        return self._a1, self._a2, self._a3

    def get_unitcell(self):
        """Return the hoomd unit cell parameter"""
        a1, a2, a3 = self.get_cell_len()
        return hoomd.lattice.unitcell(
            N=self.get_num_molecules(),
            a1=a1,
            a2=a2,
            a3=a3,
            position=self.get_positions(),
            dimensions=self.get_dimensions(),
            orientation=self.get_orientations(),
        )

    def get_matrix(self):
        return np.array([self._a1, self._a2, self._a3])

    def get_orientations(self):
        """Return the orientation quaternions of the

        Args:
            angle (float): The angle that a molecule is oriented

        Returns:
            class:`numpy.quaternion`: Quaternion representation of the angle
        """
        angles = self._orientations*(math.pi/180)
        return quaternion.as_float_array(np.array(
            [quaternion.from_euler_angles(angle, 0, 0) for angle in angles]))

    def set_positions(self, array):
        """Set the positions using absolute coordinates"""
        matrix = np.linalg.inv(np.array([self._a1, self._a2, self._a3]))
        self._positions = np.array([np.dot(pos, matrix) for pos in array])
        print("After:\n", self._positions)

    def get_num_molecules(self):
        """Return the number of molecules"""
        return len(self._orientations)


class CrysTrimer(Crystal):
    """A class for the crystal structures of the 2D Trimer molecule"""
    def __init__(self):
        super().__init__()
        self._dimensions = 2


class p2(CrysTrimer):
    """Defining the unit cell of the p2 group of the Trimer molecule"""
    def __init__(self):
        super().__init__()
        self._a1 = np.array([3.82, 0, 0])
        self._a2 = np.array([0.68, 2.53, 0])
        self._a3 = np.array([0, 0, 1])
        self._positions = np.array([[0.3, 0.32, 0], [0.7, 0.68, 0]])
        self._orientations = np.array([40, -140])


def main():
    crys = p2()
    crys.get_unitcell()
    return crys


if __name__ == "__main__":
    crys = main()
