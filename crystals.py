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
        return self._positions

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

    def get_orientations(self):
        """Return the orientation quaternions of the

        Args:
            angle (float): The angle that a molecule is oriented

        Returns:
            class:`numpy.quaternion`: Quaternion representation of the angle
        """
        angles = self._orientations*(math.pi/180)
        return quaternion.as_float_array(np.array([quaternion.from_euler_angles(angle, 0, 0)
                                                   for angle in angles]))

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
        self._a1 = [3.82, 0, 0]
        self._a2 = [0.68, 2.53, 0]
        self._a3 = [0, 0, 0]
        self._positions = [
            [0.70, 0.33, 0],
            [3.12, 2.20, 0]
        ]
        self._orientations = np.array([151, 151+180])

def main():
    crys = p2()
    crys.get_unitcell()
    return crys

if __name__ == "__main__":
    crys = main()
