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

from ..math_helper import displacement_periodic, quaternion_rotation

np.seterr(divide='raise', invalid='raise')

logger = logging.getLogger(__name__)


class dynamics(object):
    """Compute dynamic properties of a simulation."""

    dyn_dtype = np.float32

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
        self.position = position.astype(self.dyn_dtype)
        self.num_particles = position.shape[0]
        self.orientation = orientation.astype(self.dyn_dtype)

    def computeMSD(self, position: np.ndarray) -> float:
        """Compute the mean squared displacement."""
        result = translationalDisplacement(self.box, self.position, position)
        return mean_squared_displacement(result)

    def comptuteMFD(self, position: np.ndarray) -> float:
        """Comptute the fourth power of displacement."""
        result = translationalDisplacement(self.box, self.position, position)
        return mean_fourth_displacement(result)

    def computeAlpha(self, position: np.ndarray) -> float:
        r"""Compute the non-gaussian parameter alpha.

        .. math::
            \alpha = \frac{\langle \Delta r^4\rangle}
                      {2\langle \Delta r^2  \rangle^2} -1

        """
        result = translationalDisplacement(self.box, self.position, position)
        return alpha_non_gaussian(result)

    def computeTimeDelta(self, timestep: int) -> int:
        """Time difference between keyframe and timestep."""
        return timestep - self.timestep

    def computeRotation(self, orientation: np.ndarray) -> float:
        """Compute the rotation of the moleule."""
        result = rotationalDisplacement(self.orientation, orientation)
        return mean_rotation(result)

    def get_rotations(self, orientation: np.ndarray) -> np.ndarray:
        """Get all the rotations."""
        result = rotationalDisplacement(self.orientation, orientation)
        return result

    def get_displacements(self, position: np.ndarray) -> np.ndarray:
        """Get all the displacements."""
        result = translationalDisplacement(self.box, self.position, position)
        return mean_displacement(result)

    def computeAll(self,
                   timestep: int,
                   position: np.ndarray,
                   orientation: np.ndarray=None,
                   ) -> pandas.Series:
        """Compute all dynamics quantities of interest."""
        if orientation is not None:
            delta_rotation = rotationalDisplacement(self.orientation, orientation)
        else:
            delta_rotation = None

        delta_displacement = translationalDisplacement(self.box, self.position, position)
        return all_dynamics(
            self.computeTimeDelta(timestep),
            delta_displacement,
            delta_rotation,
        )

    def get_molid(self):
        """Molecule ids of each of the values."""
        return np.arange(self.num_particles)


class molecularRelaxation(object):
    """Computeteh relaxation of each molecule."""

    def __init__(self, num_elements: int, threshold: float) -> None:
        self.num_elements = num_elements
        self.threshold = threshold
        self._max_value = 2**32 - 1
        self.status = np.full(self.num_elements, self._max_value, dtype=int)

    def add(self, timediff: int, distance: np.ndarray) -> None:
        assert distance.shape == self.status.shape
        moved = np.less(self.threshold, distance)
        moveable = np.greater(self.status, timediff)
        self.status[np.logical_and(moved, moveable)] = timediff


class relaxations(object):

    def __init__(self, timestep: int,
                 box: np.ndarray,
                 position: np.ndarray,
                 orientation: np.ndarray) -> None:
        self.init_time = timestep
        self.box = box
        num_elements = position.shape[0]
        self.init_position = position
        self.init_orientation = orientation
        self.mol_relax = {
            'tau_D1': molecularRelaxation(num_elements, threshold=1.),
            'tau_D012': molecularRelaxation(num_elements, threshold=0.12),
            'tau_T2': molecularRelaxation(num_elements, threshold=np.pi/2),
            'tau_T4': molecularRelaxation(num_elements, threshold=np.pi/4),
        }

    def add(self, timestep: int,
            position: np.ndarray,
            orientation: np.ndarray,
            ) -> None:
        displacement = translationalDisplacement(self.box, self.init_position, position)
        rotation = rotationalDisplacement(self.init_orientation, orientation)
        for key, func in self.mol_relax.items():
            if 'D' in key:
                func.add(timestep, displacement)
            else:
                func.add(timestep, rotation)

    def summary(self) -> pandas.DataFrame:
        return pandas.DataFrame(
            {key: func.status for key, func in self.mol_relax.items()}
        )


def mean_squared_displacement(displacement: np.ndarray) -> float:
    """Mean value of the squared displacment.

    Args:
        displacement (class:`numpy.ndarray`): vector of squared
            displacements.

    Returns:
        float: Mean value

    """
    return np.square(displacement).mean()


def mean_fourth_displacement(displacement: np.ndarray) -> float:
    """Mean value of the fourth power of displacment.

    Args:
        displacement (class:`numpy.ndarray`): vector of squared
            displacements.

    Returns:
        float: Mean value of the fourth power

    """
    return np.power(displacement, 4).mean()


def mean_displacement(displacement: np.ndarray) -> float:
    """Mean value of the displacment.

    Args:
        displacement (class:`numpy.ndarray`): vector of squared
            displacements.

    Returns:
        float: Mean value of the displacement

    """
    return displacement.mean()


def mean_rotation(rotation: np.ndarray) -> float:
    """Mean of the rotational displacement.

    Args:
    rotation (class:`numpy.ndarray`): Vector of rotations

    Returns:
        float: Mean value of the rotation

    """
    return rotation.mean()


def alpha_non_gaussian(displacement: np.ndarray) -> float:
    r"""Compute the non-gaussian parameter :math:`\alpha`.

    The non-gaussian parameter is given as

    .. math::
        \alpha = \frac{\langle \Delta r^4\rangle}
                  {2\langle \Delta r^2  \rangle^2} -1

    Return:
        float: The non-gaussian parameter :math:`\alpha`
    """
    try:
        return (np.power(displacement, 4).mean() /
                (2 * np.square(np.square(displacement).mean()))) - 1
    except FloatingPointError:
        return 0


def structural_relax(displacement: np.ndarray,
                     dist: float=0.3) -> float:
    r"""Compute the structural relaxation.

    The structural relaxation is given as the proportion of
    particles which have moved further than `dist` from their
    initial positions.

    Args:
        displacement: displacements
        dist (float): The distance cutoff for considering relaxation.
        (defualt: 0.3)

    Return:
        float: The structural relaxation of the configuration
    """
    return np.mean(displacement < dist)


def gamma(displacement: np.ndarray,
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
    disp2 = np.square(displacement)
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


def mobile_overlap(displacement: np.ndarray,
                   rotation: np.ndarray,
                   fraction: float=0.1) -> float:
    """Find the overlap of the most mobile translators and rotators.

    This finds the proportion of molecules which reside in the top ``fraction``
    of both the rotational and translational motion.

    """
    num_elements = int(len(displacement) * fraction)
    # np.argsort will sort from smallest to largest, we are interested in the
    # largest elements so we will take from the end of the array.
    trans_order = np.argsort(displacement)[-num_elements:]
    rot_order = np.argsort(np.abs(rotation))[-num_elements:]
    return len(np.intersect1d(trans_order, rot_order)) / num_elements


def spearman_rank(displacement: np.ndarray,
                  rotation: np.ndarray,
                  fraction: float=1.) -> float:
    """Compute the Spearman Rank coefficient for fast molecules.

    This takes the molecules with the fastest 10% of the translations or
    rotations and uses this subset to compute the Spearman rank coefficient.
    """
    num_elements = int(len(displacement) * fraction)
    # np.argsort will sort from smallest to largest, we are interested in the
    # largest elements so we will take from the end of the array.
    trans_order = np.argsort(displacement)[:-num_elements-1:-1]
    rot_order = np.argsort(np.abs(rotation))[:-num_elements-1:-1]
    rho, _ = spearmanr(trans_order, rot_order)
    return rho


def all_dynamics(timediff: int,
                 displacement: np.ndarray,
                 rotation: np.ndarray=None,
                 structural_threshold: float=0.3,
                 ) -> pandas.DataFrame:
    """Compute all dynamics quantities from the base quantites.

    This computes all the possible dynamic quantities from the input arrays
    taking into account the presence of rotational data.

    Args:
        timediff (int): Time difference described by the displacement and rotation.
        displacement: (:class:`numpy.ndarray`): An array of the translational
            motion. Note that this is the distance moved, rather than the motion
            vector.
        rotations: (:class:`numpy.ndarray`): An array of the rotaional motion of
            molecules. Again note that this is the rotational displacment.

    """
    dynamic_quantities = {
        'time': timediff,
        'mean_displacement': mean_displacement(displacement),
        'msd': mean_squared_displacement(displacement),
        'mfd': mean_fourth_displacement(displacement),
        'alpha': alpha_non_gaussian(displacement),
    }
    if rotation is not None:
        dynamic_quantities.update({
            'mean_rotation': mean_rotation(rotation),
            'rot1': rotational_relax1(rotation),
            'rot2': rotational_relax2(rotation),
            'gamma': gamma(displacement, rotation),
            'spearman_rank': spearman_rank(displacement, rotation),
            'overlap': mobile_overlap(displacement, rotation),
        })
    return pandas.DataFrame(dynamic_quantities, index=[timediff])


def rotationalDisplacement(initial: np.ndarray,
                           final: np.ndarray,
                           ) -> np.ndarray:
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
    result = np.empty(final.shape[0], dtype=final.dtype)
    quaternion_rotation(initial, final, result)
    return result


def translationalDisplacement(box: np.ndarray,
                              initial: np.ndarray,
                              final: np.ndarray,
                              ) -> np.ndarray:
    """Optimised function for computing the displacement.

    This computes the displacement using the shortest path from the original
    position to the final position. This is a reasonable assumption to make
    since the path

    This assumes there is no more than a single image between molecules,
    which breaks slightly when the frame size changes. I am assuming this is
    negligible so not including it.
    """
    result = np.empty(final.shape[0], dtype=final.dtype)
    displacement_periodic(box.astype(np.float32), initial, final, result)
    return result
