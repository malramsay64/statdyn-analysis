#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Compute dynamic properties."""

import logging

import numpy as np
import pandas
from scipy.stats import spearmanr

np.seterr(divide='raise', invalid='raise')

logger = logging.getLogger(__name__)


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
        self.box = box[:3]
        self.position = position
        self.num_particles = len(position)
        self.orientation = orientation

    def computeMSD(self, position: np.ndarray) -> float:
        """Compute the mean squared displacement."""
        result = np.zeros(self.num_particles)
        squaredDisplacement(self.box, self.position, position, result)
        return mean_squared_displacement(result)

    def comptuteMFD(self, position: np.ndarray) -> float:
        """Comptute the fourth power of displacement."""
        result = np.zeros(self.num_particles)
        squaredDisplacement(self.box, self.position, position, result)
        return mean_fourth_displacement(result)

    def computeAlpha(self, position: np.ndarray) -> float:
        r"""Compute the non-gaussian parameter alpha.

        .. math::
            \alpha = \frac{\langle \Delta r^4\rangle}
                      {2\langle \Delta r^2  \rangle^2} -1

        """
        disp2 = np.empty(self.num_particles)
        squaredDisplacement(self.box, self.position, position, disp2)
        return alpha_non_gaussian(disp2)

    def computeTimeDelta(self, timestep: float) -> float:
        """Time difference between keyframe and timestep."""
        return timestep - self.timestep

    def computeRotation(self, orientation: np.ndarray) -> float:
        """Compute the rotation of the moleule."""
        result = np.empty(self.num_particles)
        rotationalDisplacement(self.orientation, orientation, result)
        return mean_rotation(result)

    def get_rotations(self, orientation: np.ndarray) -> np.ndarray:
        """Get all the rotations."""
        result = np.empty(self.num_particles)
        rotationalDisplacement(self.orientation, orientation, result)
        return result

    def get_displacements(self, position: np.ndarray) -> np.ndarray:
        """Get all the displacements."""
        result = np.empty(self.num_particles)
        squaredDisplacement(self.box, self.position, position, result)
        return mean_displacement(result)

    def computeAll(self,
                   timestep: int,
                   position: np.ndarray,
                   orientation: np.ndarray=None,
                   ) -> pandas.Series:
        """Compute all dynamics quantities of interest."""
        delta_rotation = np.empty(self.num_particles)
        rotationalDisplacement(self.orientation, orientation, delta_rotation)
        delta_displacementSq = np.empty(self.num_particles)
        squaredDisplacement(self.box, self.position, position, delta_displacementSq)
        dyn_values = all_dynamics(delta_displacementSq, delta_rotation)
        dyn_values['time'] = self.computeTimeDelta(timestep)
        return dyn_values

    def get_molid(self):
        """Molecule ids of each of the values."""
        return np.arange(self.num_particles)


def mean_squared_displacement(displacment_squared: np.ndarray) -> float:
    """Mean value of the squared displacment.

    Args:
        displacement_squared (class:`numpy.ndarray`): vector of squared
            displacements.

    Returns:
        float: Mean value

    """
    return displacment_squared.mean()


def mean_fourth_displacement(displacment_squared: np.ndarray) -> float:
    """Mean value of the fourth power of displacment.

    Args:
        displacement_squared (class:`numpy.ndarray`): vector of squared
            displacements.

    Returns:
        float: Mean value of the fourth power

    """
    return np.square(displacment_squared).mean()


def mean_displacement(displacment_squared: np.ndarray) -> float:
    """Mean value of the displacment.

    Args:
        displacement_squared (class:`numpy.ndarray`): vector of squared
            displacements.

    Returns:
        float: Mean value of the displacement

    """
    return np.sqrt(displacment_squared).mean()


def mean_rotation(rotation: np.ndarray) -> float:
    """Mean of the rotational displacement.

    Args:
    rotation (class:`numpy.ndarray`): Vector of rotations

    Returns:
        float: Mean value of the rotation

    """
    return rotation.mean()


def alpha_non_gaussian(displacement_squared: np.ndarray) -> float:
    r"""Compute the non-gaussian parameter :math:`\alpha`.

    The non-gaussian parameter is given as

    .. math::
        \alpha = \frac{\langle \Delta r^4\rangle}
                  {2\langle \Delta r^2  \rangle^2} -1

    Return:
        float: The non-gaussian parameter :math:`\alpha`
    """
    return (np.square(displacement_squared).mean() /
            (2 * np.square(displacement_squared.mean()) - 1))


def structural_relax(displacement_squared: np.ndarray,
                     dist: float=0.3) -> float:
    r"""Compute the structural relaxation.

    The structural relaxation is given as the proportion of
    particles which have moved further than `dist` from their
    initial positions.

    Args:
        dist (float): The distance cutoff for considering relaxation.
        (defualt: 0.3)

    Return:
        float: The structural relaxation of the configuration
    """
    return np.mean(displacement_squared < np.square(dist))


def gamma(displacement_squared: np.ndarray,
          rotation: np.ndarray) -> float:
    r"""Calculate the second order coupling of translations and rotations.

    .. math::
        \gamma &= \frac{\langle(\Delta r \Delta\theta)^2 \rangle -
            \langle\Delta r^2\rangle\langle\Delta \theta^2\rangle
            }{\langle\Delta r^2\rangle\langle\Delta\theta^2\rangle}

    Return:
        float: The squared coupling of translations and rotations
        :math:`\gamma`

    """
    rot2 = np.square(rotation)
    disp2 = displacement_squared
    disp2m_rot2m = disp2.mean() * rot2.mean()
    try:
        return ((disp2 * rot2).mean() - disp2m_rot2m) / disp2m_rot2m
    except FloatingPointError:
        with np.errstate(invalid='ignore'):
            res = ((disp2 * rot2).mean() - disp2m_rot2m) / disp2m_rot2m
            np.nan_to_num(res, copy=False)
            return res


def rotational_relax1(rotation: np.ndarray) -> float:
    r"""Compute the first-order rotational relaxation function.

    .. math::
        C_1(t) = \langle \hat\vec e(0) \cdot
            \hat \vec e(t) \rangle

    Return:
        float: The rotational relaxation
    """
    return np.mean(np.cos(rotation))


def rotational_relax2(rotation: np.ndarray) -> float:
    r"""Compute the second rotational relaxation function.

    .. math::
        C_1(t) = \langle 2[\hat\vec e(0) \cdot \
            \hat \vec e(t)]^2 - 1 \rangle

    Return:
        float: The rotational relaxation
    """
    return np.mean(2 * np.square(np.cos(rotation)) - 1)


def mobile_overlap(displacment_squared: np.ndarray,
                   rotation: np.ndarray,
                   fraction: float=0.1) -> float:
    """Find the overlap of the most mobile translators and rotators.

    This finds the proportion of molecules which reside in the top ``fraction``
    of both the rotational and translational motion.

    """
    num_elements = int(len(displacment_squared) * fraction)
    # np.argsort will sort from smallest to largest, we are interested in the
    # largest elements so we will take from the end of the array.
    trans_order = np.argsort(displacment_squared)[:-num_elements]
    rot_order = np.argsort(np.abs(rotation))[:-num_elements]
    return len(np.intersect1d(trans_order, rot_order)) / num_elements


def spearman_rank(displacment_squared: np.ndarray,
                  rotation: np.ndarray,
                  fraction: float=0.1) -> float:
    """Compute the Spearman Rank coefficient for fast molecules.

    This takes the molecules with the fastest 10% of the translations or
    rotations and uses this subset to compute the Spearman rank coefficient.
    """
    num_elements = int(len(displacment_squared) * fraction)
    # np.argsort will sort from smallest to largest, we are interested in the
    # largest elements so we will take from the end of the array.
    trans_order = np.argsort(displacment_squared)[:-num_elements]
    rot_order = np.argsort(np.abs(rotation))[:-num_elements]
    rho, _ = spearmanr(trans_order, rot_order)
    # Elements are in reverse order, smallest to largest so negate spearman value
    return -rho


def all_dynamics(displacement_squared: np.ndarray,
                 rotation: np.ndarray=None,
                 structural_threshold: float=0.3,
                 ) -> pandas.Series:
    """Compute all dynamics quantities from the base quantites.

    This computes all the possible dynamic quantities from the input arrays
    taking into account the presence of rotational data.

    Args:
        translations: (:class:`numpy.ndarray`): An array of the translational
            motion. Note that this is the distance moved, rather than the motion
            vector.
        rotations: (:class:`numpy.ndarray`): An array of the rotaional motion of
            molecules. Again note that this is the rotational displacment.

    """
    dynamic_quantities = {
        'mean_displacement': mean_displacement(displacement_squared),
        'msd': mean_squared_displacement(displacement_squared),
        'mfd': mean_fourth_displacement(displacement_squared),
        'alpha': alpha_non_gaussian(displacement_squared),
    }
    if rotation is not None:
        dynamic_quantities.update({
            'mean_rotation': mean_rotation(rotation),
            'rot1': rotational_relax1(rotation),
            'rot2': rotational_relax2(rotation),
            'gamma': gamma(displacement_squared, rotation),
            'spearman_rank': spearman_rank(displacement_squared, rotation, fraction=0.1),
            'overlap': mobile_overlap(displacement_squared, rotation),
        })
    return pandas.Series(dynamic_quantities)


def rotationalDisplacement(initial: np.ndarray,
                           final: np.ndarray,
                           result: np.ndarray
                           ) -> None:
    r"""Compute the rotational displacement.

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


def squaredDisplacement(box: np.ndarray,
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
