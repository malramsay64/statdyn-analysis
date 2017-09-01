#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Compute dynamic properties."""

import logging

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


class dynamics(object):
    """Compute dynamic properties of a simulation."""

    def __init__(self,
                 timestep: int,
                 box: np.ndarray,
                 position: np.ndarray,
                 orientation: np.ndarray=None,
                 ) -> None:
        """Initialise a dynamics instance.

        Args:
            timestep (int): The timestep on which the configuration was taken.
            position (py:class:`numpy.ndarray`): The positions of the molecules
                with shape ``(nmols, 3)``. Even if the simulation is only 2D,
                all 3 dimensions of the position need to be passed.
            orientation (py:class:`numpy.ndarray`): The orientaions of all the
                molecules as a quaternion in the form ``(w, x, y, z)``. If no
                orientation is supplied then no rotational quantities are
                calculated.

        """
        self.timestep = timestep
        self.box = box
        self.position = position
        self.num_particles = len(position)
        # use the quaternion library for quaternion calculations
        self.orientaion = orientation

    def computeMSD(self, position: np.ndarray) -> float:
        """Compute the mean squared displacement."""
        result = np.zeros(self.num_particles)
        squaredDisplacment(self.box, self.position, position, result)
        return result.mean()

    def comptuteMFD(self, position: np.ndarray) -> float:
        """Comptute the fourth power of displacement."""
        result = np.zeros(self.num_particles)
        squaredDisplacment(self.box, self.position, position, result)
        return np.square(result).mean()

    def computeAlpha(self, position: np.ndarray) -> float:
        r"""Compute the non-gaussian parameter alpha.

        .. math::
            \alpha = \frac{\langle \Delta r^4\rangle}
                      {2\langle \Delta r^2  \rangle^2} -1

        """
        disp2 = np.zeros(self.num_particles)
        squaredDisplacment(self.box, self.position, position, disp2)
        return (np.square(disp2).mean() / (2*np.square(disp2).mean())) - 1

    def computeRotation(self, orientation: np.ndarray) -> float:
        """Compute the rotation of the moleule."""
        result = np.zeros(self.num_particles)
        if self.orientaion:
            rotationalDisplacement(self.orientaion, orientation, result)
        return result


def rotationalDisplacement(initial: np.ndarray,
                           final: np.ndarray,
                           result: np.ndarray
                           ) -> None:
    """Compute the rotational displacement.

    Args:
        initial (py:class:`numpy.ndarray`): Initial orientation.
        final (py:class:`numpy.ndarray`): final orientation.
        result (py:class:`numpy.ndarray`): array in which to store result

    The rotational displacment is computed using a slightly modified formula
    from [@Huynh2009]_ specifically the formula for :math:`\phi_3`. Since we
    are interested in angles of the range :math:`[0, 2\pi]`, the result of
    :math:`\phi_3` is multiplied by 2, which is shown by Huynh to be equal
    to :math:`phi_6`.

    This imlementation was chosen for speed and accuracy, being tested against
    a number of other possibilities. Another notable formulation was by [Jim
    Belk] on Stack Exchange, however this equation was both slower to compute
    in addition to being more prone to unusual bnehaviour.

    .. [@Hunyh2009]: 1. Huynh, D. Q. Metrics for 3D rotations: Comparison and
        analysis.  J. Math. Imaging Vis. 35, 155–164 (2009).
    .. [Jim Belk]: https://math.stackexchange.com/questions/90081/quaternion-distance

    """
    with np.errstate(invalid='ignore'):
        result[:] = 2*np.arccos(np.abs(np.einsum('ij,ij->i', initial, final)))
    np.nan_to_num(result, copy=False)


def squaredDisplacment(box: np.ndarray,
                       initial: np.ndarray,
                       final: np.ndarray,
                       result: np.ndarray
                       ) -> None:
    """Optimised function for computing the squared displacement.

    This computes the displacment using the shortest path from the original
    position to the final position. This is a reasonable assumption to make
    since the path
    """
    box_sq = np.square(box)
    temp = np.square(initial - final)
    periodic = np.where(box_sq < temp)
    # Periodic contains 2 numpy arrays, one indexing each dimension. Here I am
    # taking the position in the second dimension which indicates the box
    # dimension and subtracting from the result to give the periodic distance.
    temp[periodic] -= box_sq[periodic[1]]
    result[:] = temp.sum(axis=1)
