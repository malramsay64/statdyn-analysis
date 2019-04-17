#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Compute dynamic properties."""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas
from freud.box import Box

from .frame import Frame
from .molecules import Molecule
from .util import create_freud_box, quaternion_rotation, rotate_vectors

np.seterr(divide="raise", invalid="raise", over="raise")
logger = logging.getLogger(__name__)

YamlValue = Union[str, float, int]


class Dynamics:
    """Compute dynamic properties of a simulation.

    Args:
        timestep: The timestep on which the configuration was taken.
        box: The lengths of each side of the simulation cell including
            any tilt factors.
        position: The positions of the molecules
            with shape ``(nmols, 3)``. Even if the simulation is only 2D,
            all 3 dimensions of the position need to be passed.
        orientation: The orientaions of all the
            molecules as a quaternion in the form ``(w, x, y, z)``. If no
            orientation is supplied then no rotational quantities are
            calculated.
        molecule: The molecule for which to compute the dynamics quantities.
            This is used to compute the structural relaxation for all particles.
        image: The periodic image each particle is occupying.
        wave_number: The wave number of the maximum peak in the Fourier
            transform of the radial distribution function. If None this is
            calculated from the initial configuration.

    """

    def __init__(
        self,
        timestep: int,
        box: np.ndarray,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        molecule: Optional[Molecule] = None,
        image: Optional[np.ndarray] = None,
        wave_number: Optional[float] = None,
    ) -> None:
        if position.shape[0] == 0:
            raise RuntimeError("Position must contain values, has length of 0.")
        if molecule is None:
            is2D = True
        else:
            is2D = molecule.dimensions == 2

        self.box = create_freud_box(box, is_2D=is2D)

        self.timestep = timestep
        self.position = position
        self.num_particles = position.shape[0]
        if orientation is not None:
            if orientation.shape[0] == 0:
                raise RuntimeError("Orientation must contain values, has length of 0.")
            self.orientation = orientation
        if molecule is not None:
            self.mol_vector = molecule.positions
        self.image = image

        if wave_number is None:
            logger.info("No wave number passed, calculating from current frame.")
            raise NotImplementedError(
                "Automatic calculation of wave-number is not yet supported."
            )

        self.wave_number = wave_number

    @classmethod
    def from_frame(
        cls,
        frame: Frame,
        molecule: Optional[Molecule] = None,
        wave_number: Optional[float] = None,
    ) -> "Dynamics":
        """Initialise the Dynamics class from a Frame object.

        There is significant overlap between the frame class and the dynamics class,
        so this is a convenience method to make the initialisation simpler.

        """
        return cls(
            frame.timestep,
            frame.box,
            frame.position,
            frame.orientation,
            molecule=molecule,
            image=frame.image,
            wave_number=wave_number,
        )

    def compute_msd(
        self, position: np.ndarray, image: Optional[np.ndarray] = None
    ) -> float:
        """Compute the mean squared displacement."""
        result = translational_displacement(
            self.box, self.position, position, self.image, image
        )
        return mean_squared_displacement(result)

    def compute_mfd(
        self, position: np.ndarray, image: Optional[np.ndarray] = None
    ) -> float:
        """Compute the fourth power of displacement."""
        result = translational_displacement(
            self.box, self.position, position, self.image, image
        )
        return mean_fourth_displacement(result)

    def compute_alpha(
        self, position: np.ndarray, image: Optional[np.ndarray] = None
    ) -> float:
        r"""Compute the non-gaussian parameter alpha.

        .. math::
            \alpha = \frac{\langle \Delta r^4\rangle}
                      {2\langle \Delta r^2  \rangle^2} -1

        """
        result = translational_displacement(
            self.box, self.position, position, self.image, image
        )
        return alpha_non_gaussian(result)

    def compute_time_delta(self, timestep: int) -> int:
        """Time difference between keyframe and timestep."""
        return timestep - self.timestep

    def compute_rotation(self, orientation: np.ndarray) -> float:
        """Compute the rotation from the initial frame."""
        result = rotational_displacement(self.orientation, orientation)
        return mean_rotation(result)

    def get_rotations(self, orientation: np.ndarray) -> np.ndarray:
        """Get all the rotations."""
        result = rotational_displacement(self.orientation, orientation)
        return result

    def get_displacements(
        self, position: np.ndarray, image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get all the displacements."""
        result = translational_displacement(
            self.box, self.position, position, self.image, image
        )
        return result

    def compute_struct_relax(
        self, position: np.ndarray, orientation: np.ndarray, threshold: float = 0.3
    ) -> float:
        particle_displacement = translational_displacement(
            self.box,
            molecule2particles(self.position, self.orientation, self.mol_vector),
            molecule2particles(position, orientation, self.mol_vector),
        )
        return structural_relax(particle_displacement, threshold)

    def compute_all(
        self,
        timestep: int,
        position: np.ndarray,
        orientation: np.ndarray = None,
        image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute all dynamics quantities of interest."""
        delta_displacement = translational_displacement(
            self.box, self.position, position, self.image, image
        )
        dynamic_quantities = {
            "time": self.compute_time_delta(timestep),
            "mean_displacement": mean_displacement(delta_displacement),
            "msd": mean_squared_displacement(delta_displacement),
            "mfd": mean_fourth_displacement(delta_displacement),
            "alpha": alpha_non_gaussian(delta_displacement),
            "com_struct": structural_relax(
                delta_displacement, dist=np.pi / (2 * self.wave_number)
            ),
            "scattering_function": intermediate_scattering_function(
                self.box, self.position, position, wave_number=self.wave_number
            ),
        }
        if self.orientation is not None:
            delta_rotation = rotational_displacement(self.orientation, orientation)
            dynamic_quantities.update(
                {
                    "mean_rotation": mean_rotation(delta_rotation),
                    "rot1": rotational_relax1(delta_rotation),
                    "rot2": rotational_relax2(delta_rotation),
                    "gamma": gamma(delta_displacement, delta_rotation),
                    "overlap": mobile_overlap(delta_displacement, delta_rotation),
                    "struct": self.compute_struct_relax(
                        position, orientation, threshold=np.pi / (2 * self.wave_number)
                    ),
                }
            )
        return dynamic_quantities

    def __len__(self) -> int:
        return self.num_particles

    def get_molid(self):
        """Molecule ids of each of the values."""
        return np.arange(self.num_particles)


class MolecularRelaxation:
    """Compute the relaxation of each molecule."""

    def __init__(self, num_elements: int, threshold: float) -> None:
        self.num_elements = num_elements
        self.threshold = threshold
        self._max_value = 2 ** 32 - 1
        self._status = np.full(self.num_elements, self._max_value, dtype=int)

    def add(self, timediff: int, distance: np.ndarray) -> None:
        if distance.shape != self._status.shape:
            raise RuntimeError(
                "Current state and initial state have different shapes. "
                "current: {distance.shape}, initial: {self._status.shape}"
            )
        with np.errstate(invalid="ignore"):
            moved = np.greater(distance, self.threshold)
            moveable = np.greater(self._status, timediff)
            self._status[np.logical_and(moved, moveable)] = timediff

    def get_status(self):
        return self._status


class LastMolecularRelaxation(MolecularRelaxation):
    _is_irreversible = 3

    def __init__(
        self, num_elements: int, threshold: float, irreversibility: float = 1.0
    ) -> None:
        super().__init__(num_elements, threshold)
        self._state = np.zeros(self.num_elements, dtype=np.uint8)
        self._irreversibility = irreversibility

    def add(self, timediff: int, distance: np.ndarray) -> None:
        if distance.shape != self._status.shape:
            raise RuntimeError(
                "Current state and initial state have different shapes. "
                "current: {distance.shape}, initial: {self._status.shape}"
            )

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


class StructRelaxations(MolecularRelaxation):
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
    last_passage_cutoff: Optional[float] = None,
) -> MolecularRelaxation:

    if threshold is None or threshold < 0:
        raise ValueError(f"Threshold needs a positive value, got {threshold}")

    if last_passage:
        if last_passage_cutoff is None:
            logger.info(
                "No last passage cutoff given, using 3 * threshold: %f", 3 * threshold
            )
            last_passage_cutoff = 3 * threshold
        elif last_passage_cutoff < 0:
            raise ValueError(
                "When using last passage a positive cutoff value is required,"
                f"got {last_passage_cutoff}, default is 1.0"
            )

        return LastMolecularRelaxation(num_elements, threshold, last_passage_cutoff)

    return MolecularRelaxation(num_elements, threshold)


class Relaxations:
    def __init__(
        self,
        timestep: int,
        box: np.ndarray,
        position: np.ndarray,
        orientation: np.ndarray,
        molecule: Optional[Molecule] = None,
        is2D: Optional[bool] = None,
        wave_number: Optional[float] = None,
    ) -> None:
        self.init_time = timestep
        if molecule is None:
            is2D = False
        else:
            is2D = molecule.dimensions == 2

        self.box = create_freud_box(box, is_2D=is2D)

        self._num_elements = position.shape[0]
        self.init_position = position
        self.init_orientation = orientation
        if wave_number is None:
            raise NotImplementedError(
                "Automatic calculation of wave number is not yet supported."
            )
        self.wave_number = wave_number

        # set defualt values for mol_relax
        self.set_mol_relax(
            [
                {"name": "tau_D", "threshold": 3 * wave_number},
                {"name": "tau_F", "threshold": wave_number},
                {"name": "tau_L", "threshold": wave_number, "last_passage": True},
                {"name": "tau_T2", "threshold": np.pi / 2},
                {"name": "tau_T3", "threshold": np.pi / 3},
                {"name": "tau_T4", "threshold": np.pi / 4},
            ]
        )

    @classmethod
    def from_frame(
        cls,
        frame: Frame,
        molecule: Optional[Molecule] = None,
        wave_number: Optional[float] = None,
    ) -> "Relaxations":
        """Initialise a Relaxations class from a Frame class.

        This uses the properties of the Frame class to fill the values of the
        Relaxations class, for which there is significant overlap.

        """
        return cls(
            frame.timestep,
            frame.box,
            frame.position,
            frame.orientation,
            molecule=molecule,
            wave_number=wave_number,
        )

    def set_mol_relax(self, definition: List[Dict[str, YamlValue]]) -> None:
        self.mol_relax: Dict[str, MolecularRelaxation] = {}
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
        displacement = translational_displacement(
            self.box, self.init_position, position
        )
        rotation = rotational_displacement(self.init_orientation, orientation)
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
    if np.allclose(orientation, 0.0):
        orientation[:, 0] = 1.0
    return np.concatenate(
        [rotate_vectors(orientation, pos) for pos in mol_vector.astype(np.float32)]
    ) + np.repeat(position, mol_vector.shape[0], axis=0)


# This decorator is what enables the caching of this function,
# making this function 100 times faster for subsequent exectutions
@lru_cache()
def create_wave_vector(wave_number: float, angular_resolution: int):
    r"""Convert a wave number into a radially symmetric wave vector

    This calculates the values of cos and sin :math:`\theta` for `angular_resolution`
    values of :math:`\theta` between 0 and :math:`2\pi`.

    The results of this function are cached, so these values only need to be computed
    a single time, the rest of the time they are just returned.

    """
    angles = np.linspace(0, 2 * np.pi, num=angular_resolution, endpoint=False).reshape(
        (-1, 1)
    )
    wave_vector = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
    return wave_vector * wave_number


def intermediate_scattering_function(
    box: Box,
    initial_position: np.ndarray,
    current_position: np.ndarray,
    wave_number: float,
    angular_resolution: int = 60,
) -> float:
    r"""Calculate the intermediate scattering function for a specific wave-vector

    This calculates the equation

    .. math::
        F(k, t) = \langle \cos( k [r_{x}(0) - r_{x}(t)]) \rangle

    Where k is the value of `wave_vector`, the values of the array `inital_position` are $r_x(0)$,
    while `current_position` is $r_(t)$.

    The values of initial_position and current_position are both expected to be a vector of
    shape N x 3 and the appropriate elementn are extracted from it.

    """
    wave_vector = create_wave_vector(wave_number, angular_resolution)

    displacement = box.wrap(initial_position - current_position)[:, :2]

    return np.mean(np.cos(np.dot(wave_vector, displacement.T)))


def mean_squared_displacement(displacement: np.ndarray) -> float:
    """Mean value of the squared displacement.

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
        dist): The distance cutoff for considering relaxation. (defualt: 0.3)

    Return:
        float: The structural relaxation of the configuration
    """
    try:
        return np.mean(displacement < dist)
    except FloatingPointError:
        return np.nan


def gamma(displacement: np.ndarray, rotation: np.ndarray) -> float:
    r"""Calculate the second order coupling of translations and rotations.

    .. math::
        \gamma = \frac{\langle(\Delta r \Delta\theta)^2 \rangle -
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
        C_1(t) = \langle \hat{\mathbf{e}}(0) \cdot \hat{\mathbf{e}}(t) \rangle

    Return:
        float: The rotational relaxation
    """
    return np.mean(np.cos(rotation))


def rotational_relax2(rotation: np.ndarray) -> float:
    r"""Compute the second rotational relaxation function.

    .. math::
        C_1(t) = \langle 2(\hat{\mathbf{e}}(0) \cdot  \hat{\mathbf{e}}(t))^2 - 1 \rangle

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


def rotational_displacement(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
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
    if final.shape[0] == 0:
        raise RuntimeError("final must contain values, has length of 0.")

    if initial.shape != final.shape:
        raise RuntimeError(
            "Final state and initial state have different shapes. "
            "initial: {initial.shape}, final: {final.shape}"
        )

    return quaternion_rotation(initial, final)


# pylint: disable=unused-argument
def translational_displacement(
    box: Box,
    initial: np.ndarray,
    final: np.ndarray,
    initial_image: Optional[np.ndarray] = None,
    final_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Optimised function for computing the displacement."""
    if final.shape[0] == 0:
        raise RuntimeError("final must contain values, has length of 0.")

    if initial.shape != final.shape:
        raise RuntimeError(
            "Final state and initial state have different shapes. "
            "initial: {initial.shape}, final: {final.shape}"
        )

    if not isinstance(box, Box):
        raise ValueError(f"Expecting type of {Box}, got {type(box)}")

    try:
        return np.linalg.norm(box.wrap(final - initial), axis=1)
    except FloatingPointError:
        return np.full(initial.shape[0], np.nan)
