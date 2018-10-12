#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Compute dynamic properties."""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas
from freud.box import Box

from .molecules import Molecule, Trimer
from .util import displacement_periodic, quaternion_rotation, rotate_vectors

np.seterr(divide="raise", invalid="raise", over="raise")
logger = logging.getLogger(__name__)

YamlValue = Union[str, float, int]


class dynamics:
    """Compute dynamic properties of a simulation."""

    def __init__(
        self,
        timestep: int,
        box: np.ndarray,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        molecule: Molecule = Trimer(),
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
        assert position.shape[0] > 0
        if molecule is None:
            is2D = False
        else:
            is2D = True if molecule.dimensions == 2 else False

        self.timestep = timestep
        self.box = Box(*box, is2D=is2D)
        self.position = position
        self.num_particles = position.shape[0]
        if orientation is not None:
            assert orientation.shape[0] > 0
            self.orientation = orientation
        self.mol_vector = molecule.positions

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
        return result

    def computeStructRelax(
        self, position: np.ndarray, orientation: np.ndarray, threshold: float = 0.3
    ) -> float:
        particle_displacement = translationalDisplacement(
            self.box,
            molecule2particles(self.position, self.orientation, self.mol_vector),
            molecule2particles(position, orientation, self.mol_vector),
        )
        return structural_relax(particle_displacement, threshold)

    def computeAll(
        self, timestep: int, position: np.ndarray, orientation: np.ndarray = None
    ) -> Dict[str, Any]:
        """Compute all dynamics quantities of interest."""
        delta_displacement = translationalDisplacement(
            self.box, self.position, position
        )
        dynamic_quantities = {
            "time": self.computeTimeDelta(timestep),
            "mean_displacement": mean_displacement(delta_displacement),
            "msd": mean_squared_displacement(delta_displacement),
            "mfd": mean_fourth_displacement(delta_displacement),
            "alpha": alpha_non_gaussian(delta_displacement),
            "com_struct": structural_relax(delta_displacement, dist=0.4),
        }
        if self.orientation is not None:
            delta_rotation = rotationalDisplacement(self.orientation, orientation)
            dynamic_quantities.update(
                {
                    "mean_rotation": mean_rotation(delta_rotation),
                    "rot1": rotational_relax1(delta_rotation),
                    "rot2": rotational_relax2(delta_rotation),
                    "gamma": gamma(delta_displacement, delta_rotation),
                    "overlap": mobile_overlap(delta_displacement, delta_rotation),
                    "struct": self.computeStructRelax(
                        position, orientation, threshold=0.3
                    ),
                }
            )
        return dynamic_quantities

    def get_molid(self):
        """Molecule ids of each of the values."""
        return np.arange(self.num_particles)


class molecularRelaxation:
    """Compute the relaxation of each molecule."""

    def __init__(self, num_elements: int, threshold: float) -> None:
        self.num_elements = num_elements
        self.threshold = threshold
        self._max_value = 2 ** 32 - 1
        self._status = np.full(self.num_elements, self._max_value, dtype=int)

    def add(self, timediff: int, distance: np.ndarray) -> None:
        assert distance.shape == self._status.shape
        with np.errstate(invalid="ignore"):
            moved = np.greater(distance, self.threshold)
            moveable = np.greater(self._status, timediff)
            self._status[np.logical_and(moved, moveable)] = timediff

    def get_status(self):
        return self._status


class lastMolecularRelaxation(molecularRelaxation):
    _is_irreversible = 3

    def __init__(
        self, num_elements: int, threshold: float, irreversibility: float = 1.
    ) -> None:
        super().__init__(num_elements, threshold)
        self._state = np.zeros(self.num_elements, dtype=np.uint8)
        self._irreversibility = irreversibility

    def add(self, timediff: int, distance: np.ndarray) -> None:
        assert distance.shape == self._status.shape
        with np.errstate(invalid="ignore"):
            state = np.greater(distance, self.threshold).astype(np.uint8)
            state[
                np.logical_or(
                    self._state == self._is_irreversible,
                    np.greater(distance, self._irreversibility),
                )
            ] = self._is_irreversible
            self._status[np.logical_and(state == 1, self._state == 0)] = timediff
            self._state = state

    def get_status(self):
        status = np.copy(self._status)
        status[self._state != self._is_irreversible] = self._max_value
        return status


class structRelaxations(molecularRelaxation):
    """Compute the average structural relaxation for a molecule."""

    def __init__(self, num_elements: int, threshold: float, molecule: Molecule) -> None:
        self.molecule = molecule
        super().__init__(num_elements * self.molecule.num_particles, threshold)

    def get_status(self):
        return self._status.reshape((-1, self.molecule.num_particles)).mean(axis=1)


def create_mol_relaxations(
    num_elements: int,
    threshold: float,
    last_passage: bool = False,
    last_passage_cutoff: float = 1.0,
) -> molecularRelaxation:
    if threshold is None or threshold < 0:
        raise ValueError(f"Threshold needs a positive value, got {threshold}")
    if last_passage:
        if last_passage_cutoff is None or last_passage_cutoff < 0:
            raise ValueError(
                "When using last passage a positive cutoff value is required,"
                f"got {last_passage_cutoff}, default is 1.0"
            )
        return lastMolecularRelaxation(num_elements, threshold, last_passage_cutoff)

    return molecularRelaxation(num_elements, threshold)


class relaxations:
    def __init__(
        self,
        timestep: int,
        box: np.ndarray,
        position: np.ndarray,
        orientation: np.ndarray,
        molecule: Molecule = None,  # pylint: disable=unused-argument
        is2D: bool = None,
    ) -> None:
        self.init_time = timestep
        if molecule is None:
            is2D = False
        else:
            is2D = True if molecule.dimensions == 2 else False
        self.box = Box(*box, is2D=is2D)
        self._num_elements = position.shape[0]
        self.init_position = position
        self.init_orientation = orientation
        # set defualt values for mol_relax
        self.set_mol_relax(
            [
                {"name": "tau_D1", "threshold": 1.},
                {"name": "tau_D04", "threshold": 0.4},
                {"name": "tau_DL04", "threshold": 0.4, "last_passage": True},
                {"name": "tau_T2", "threshold": np.pi / 2},
                {"name": "tau_T3", "threshold": np.pi / 3},
                {"name": "tau_T4", "threshold": np.pi / 4},
            ]
        )

    def set_mol_relax(self, definition: List[Dict[str, YamlValue]]) -> None:
        self.mol_relax: Dict[str, molecularRelaxation] = {}
        for item in definition:
            if item.get("name") is None:
                raise ValueError("'name' is a required attribute")
            index = str(item["name"])
            if item.get("threshold") is None:
                raise ValueError("'threshold' is a required attribute")
            threshold = float(item["threshold"])
            last_passage = bool(item.get("last_passage", False))
            last_passage_cutoff = float(item.get("last_passage_cutoff", 1.0))

            self.mol_relax[index] = create_mol_relaxations(
                self._num_elements,
                threshold=threshold,
                last_passage=last_passage,
                last_passage_cutoff=last_passage_cutoff,
            )

    def get_timediff(self, timestep: int):
        return timestep - self.init_time

    def add(self, timestep: int, position: np.ndarray, orientation: np.ndarray) -> None:
        displacement = translationalDisplacement(self.box, self.init_position, position)
        rotation = rotationalDisplacement(self.init_orientation, orientation)
        for key, func in self.mol_relax.items():
            if "D" in key:
                func.add(self.get_timediff(timestep), displacement)
            else:
                func.add(self.get_timediff(timestep), rotation)

    def summary(self) -> pandas.DataFrame:
        return pandas.DataFrame(
            {key: func.get_status() for key, func in self.mol_relax.items()}
        )


def molecule2particles(
    position: np.ndarray, orientation: np.ndarray, mol_vector: np.ndarray
) -> np.ndarray:
    if np.allclose(orientation, 0.):
        orientation[:, 0] = 1.
    return np.concatenate(
        [rotate_vectors(orientation, pos) for pos in mol_vector.astype(np.float32)]
    ) + np.repeat(position, mol_vector.shape[0], axis=0)


def mean_squared_displacement(displacement: np.ndarray) -> float:
    """Mean value of the squared displacment.

    Args:
        displacement (class:`numpy.ndarray`): vector of squared
            displacements.

    Returns:
        float: Mean value

    """
    if displacement.shape[0] > 0:
        return np.square(displacement).mean()
    logger.info("displacement has shape: %s", displacement.shape)
    return 0


def mean_fourth_displacement(displacement: np.ndarray) -> float:
    """Mean value of the fourth power of displacment.

    Args:
        displacement (class:`numpy.ndarray`): vector of squared
            displacements.

    Returns:
        float: Mean value of the fourth power

    """
    if displacement.shape[0] > 0:
        return np.power(displacement, 4).mean()
    logger.info("displacement has shape: %s", displacement.shape)
    return 0


def mean_displacement(displacement: np.ndarray) -> float:
    """Mean value of the displacment.

    Args:
        displacement (class:`numpy.ndarray`): vector of squared
            displacements.

    Returns:
        float: Mean value of the displacement

    """
    if displacement.shape[0] > 0:
        return displacement.mean()
    logger.info("displacement has shape: %s", displacement.shape)
    return 0


def mean_rotation(rotation: np.ndarray) -> float:
    """Mean of the rotational displacement.

    Args:
    rotation (class:`numpy.ndarray`): Vector of rotations

    Returns:
        float: Mean value of the rotation

    """
    if rotation.shape[0] > 0:
        return rotation.mean()
    logger.info("displacement has shape: %s", rotation.shape)
    return 0


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
        return (
            np.power(displacement, 4).mean()
            / (2 * np.square(np.square(displacement).mean()))
        ) - 1

    except FloatingPointError:
        return 0


def structural_relax(displacement: np.ndarray, dist: float = 0.3) -> float:
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
    try:
        return np.mean(displacement < dist)
    except FloatingPointError:
        return np.full_like(displacement, np.nan)


def gamma(displacement: np.ndarray, rotation: np.ndarray) -> float:
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
        with np.errstate(invalid="ignore"):
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


def mobile_overlap(
    displacement: np.ndarray, rotation: np.ndarray, fraction: float = 0.1
) -> float:
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


def rotationalDisplacement(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
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
    assert final.shape[0] > 0
    assert initial.shape[0] >= final.shape[0]
    return quaternion_rotation(initial, final)


def translationalDisplacement(
    box: Box, initial: np.ndarray, final: np.ndarray
) -> np.ndarray:
    """Optimised function for computing the displacement.

    This computes the displacement using the shortest path from the original
    position to the final position. This is a reasonable assumption to make
    since the path

    This assumes there is no more than a single image between molecules,
    which breaks slightly when the frame size changes. I am assuming this is
    negligible so not including it.
    """
    assert final.shape[0] > 0
    assert initial.shape[0] >= final.shape[0]
    return displacement_periodic(box, initial, final)
