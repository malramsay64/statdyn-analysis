#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Compute dynamic properties."""

import logging
from typing import Dict, Optional, Union

import numpy as np
import rowan

from ..frame import Frame
from ..molecules import Molecule
from ..util import create_freud_box, zero_quaternion
from ._util import TrackedMotion, molecule2particles

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
        orientation: The orientations of all the
            molecules as a quaternion in the form ``(w, x, y, z)``. If no
            orientation is supplied then no rotational quantities are
            calculated.
        molecule: The molecule for which to compute the dynamics quantities.
            This is used to compute the structural relaxation for all particles.
        wave_number: The wave number of the maximum peak in the Fourier
            transform of the radial distribution function. If None this is
            calculated from the initial configuration.

    """

    _all_quantities = [
        "time",
        "mean_displacement",
        "msd",
        "mfd",
        "alpha",
        "scattering_function",
        "com_struct",
        "mean_rotation",
        "msr",
        "rot1",
        "rot2",
        "alpha_rot",
        "gamma",
        "overlap",
        "struct",
    ]

    def __init__(
        self,
        timestep: int,
        box: np.ndarray,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        molecule: Optional[Molecule] = None,
        wave_number: Optional[float] = None,
        angular_resolution=360,
    ) -> None:
        """Initialise Dynamics.

        Args:
            timestep: The timestep at which the reference frame was created.
            box: The lengths of the simulation cell
            position: The initial position of each particle.
            orientation: The initial orientation of each particle.
            molecule:  The molecule which is represented by each position and orientation.
            wave_number: The wave number corresponding to the maximum in the
                static structure factor.
            angular_resolution: The angular resolution of the intermediate scattering function.

        """
        if position.shape[0] == 0:
            raise RuntimeError("Position must contain values, has length of 0.")
        if molecule is None:
            is2d = True
            self.mol_vector = None
        else:
            is2d = molecule.dimensions == 2
            self.mol_vector = molecule.positions

        self.motion = TrackedMotion(
            create_freud_box(box, is_2D=is2d), position, orientation
        )
        self.timestep = timestep
        self.num_particles = position.shape[0]

        self.wave_number = wave_number
        if self.wave_number is not None:
            angles = np.linspace(
                0, 2 * np.pi, num=angular_resolution, endpoint=False
            ).reshape((-1, 1))
            self.wave_vector = (
                np.concatenate([np.cos(angles), np.sin(angles)], axis=1) * wave_number
            )

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
            wave_number=wave_number,
        )

    @property
    def delta_translation(self):
        return self.motion.delta_translation

    @property
    def delta_rotation(self):
        return self.motion.delta_rotation

    def add(self, position: np.ndarray, orientation: Optional[np.ndarray] = None):
        """Update the state of the dynamics calculations by adding a Frame.

        This updates the motion of the particles, comparing the positions and
        orientations of the current frame with the previous frame, adding the difference
        to the total displacement. This approach allows for tracking particles over
        periodic boundaries, or through larger rotations assuming that there are
        sufficient frames to capture the information. Each single displacement obeys the
        minimum image convention, so for large time intervals it is still possible to
        have missing information.

        Args:
            frame: The configuration containing the current particle information.

        """
        self.motion.add(position, orientation)

    def add_frame(self, frame: Frame):
        """Update the state of the dynamics calculations by adding a Frame.

        This updates the motion of the particles, comparing the positions and
        orientations of the current frame with the previous frame, adding the difference
        to the total displacement. This approach allows for tracking particles over
        periodic boundaries, or through larger rotations assuming that there are
        sufficient frames to capture the information. Each single displacement obeys the
        minimum image convention, so for large time intervals it is still possible to
        have missing information.

        Args:
            frame: The configuration containing the current particle information.

        """
        self.motion.add(frame.position, frame.orientation)

    def compute_displacement(self) -> np.ndarray:
        """Compute the translational displacement for each particle."""
        return np.linalg.norm(self.delta_translation, axis=1)

    def compute_displacement2(self) -> np.ndarray:
        """Compute the squared displacement for each particle."""
        return np.square(self.delta_translation).sum(axis=1)

    def compute_msd(self) -> float:
        """Compute the mean squared displacement."""
        return self.compute_displacement2().mean()

    def compute_mfd(self) -> float:
        """Compute the fourth power of displacement."""
        return np.square(self.compute_displacement2()).mean()

    def compute_alpha(self) -> float:
        r"""Compute the non-Gaussian parameter alpha for translational motion.

        .. math::
            \alpha = \frac{\langle \Delta r^4\rangle}
                      {2\langle \Delta r^2  \rangle^2} -1

        """
        disp2 = self.compute_displacement2()
        try:
            return np.square(disp2).mean() / (2 * np.square((disp2).mean())) - 1

        except FloatingPointError:
            with np.errstate(invalid="ignore"):
                res = np.square(disp2).mean() / (2 * np.square((disp2).mean())) - 1
                np.nan_to_num(res, copy=False)
                return res

    def compute_time_delta(self, timestep: int) -> int:
        """Time difference between initial frame and timestep."""
        return timestep - self.timestep

    def compute_rotation(self) -> np.ndarray:
        """Compute the rotational motion for each particle."""
        return np.linalg.norm(self.delta_rotation, axis=1)

    def compute_rotation2(self) -> np.ndarray:
        """Compute the rotation from the initial frame."""
        return np.square(self.delta_rotation).sum(axis=1)

    def compute_mean_rotation(self) -> float:
        """Compute the rotation from the initial frame."""
        return self.compute_rotation().mean()

    def compute_msr(self) -> float:
        """Compute the mean squared rotation from the initial frame."""
        return self.compute_rotation2().mean()

    def compute_isf(self) -> float:
        """Compute the intermediate scattering function."""
        return np.cos(np.dot(self.wave_vector, self.delta_translation[:, :2].T)).mean()

    def compute_rotational_relax1(self) -> float:
        r"""Compute the first-order rotational relaxation function.

        .. math::
            C_1(t) = \langle \hat{\mathbf{e}}(0) \cdot \hat{\mathbf{e}}(t) \rangle

        Return:
            float: The rotational relaxation

        """
        return np.cos(self.compute_rotation()).mean()

    def compute_rotational_relax2(self) -> float:
        r"""Compute the second rotational relaxation function.

        .. math::
            C_1(t) = \langle 2(\hat{\mathbf{e}}(0) \cdot \hat{\mathbf{e}}(t))^2 - 1 \rangle

        Return:
            float: The rotational relaxation

        """
        return np.mean(2 * np.square(np.cos(self.compute_rotation())) - 1)

    def compute_alpha_rot(self) -> float:
        r"""Compute the non-Gaussian parameter alpha for rotational motion.

        .. math::
            \alpha = \frac{\langle \Delta \theta^4\rangle}
                      {2\langle \Delta \theta^2  \rangle^2} -1

        """
        theta2 = self.compute_rotation2()
        try:
            return np.square(theta2).mean() / (2 * np.square((theta2).mean())) - 1

        except FloatingPointError:
            with np.errstate(invalid="ignore"):
                res = np.square(theta2).mean() / (2 * np.square((theta2).mean())) - 1
                np.nan_to_num(res, copy=False)
                return res

    def compute_gamma(self) -> float:
        r"""Calculate the second order coupling of translations and rotations.

        .. math::
            \gamma = \frac{\langle(\Delta r \Delta\theta)^2 \rangle}
                {\langle\Delta r^2\rangle\langle\Delta\theta^2\rangle} - 1

        Return:
            float: The squared coupling of translations and rotations
            :math:`\gamma`

        """
        theta2 = self.compute_rotation2()
        disp2 = self.compute_displacement2()
        try:
            return ((disp2 * theta2).mean()) / (disp2.mean() * theta2.mean()) - 1

        except FloatingPointError:
            with np.errstate(invalid="ignore"):
                res = ((disp2 * theta2).mean()) / (disp2.mean() * theta2.mean()) - 1
                np.nan_to_num(res, copy=False)
                return res

    def compute_struct_relax(self) -> float:
        if self.distance is None:
            raise ValueError(
                "The wave number is required for the structural relaxation."
            )
        quat_rot = rowan.from_euler(
            self.delta_rotation[:, 2],
            self.delta_rotation[:, 1],
            self.delta_rotation[:, 0],
        )
        final_pos = molecule2particles(
            self.delta_translation, quat_rot, self.mol_vector
        )
        init_pos = molecule2particles(
            np.zeros_like(self.delta_translation),
            zero_quaternion(self.num_particles),
            self.mol_vector,
        )
        try:
            return np.mean(np.linalg.norm(final_pos - init_pos, axis=1) < self.distance)
        except FloatingPointError:
            return np.nan

    def compute_all(
        self,
        timestep: int,
        position: np.ndarray,
        orientation: np.ndarray = None,
        scattering_function: bool = False,
    ) -> Dict[str, Union[int, float]]:
        """Compute all possible dynamics quantities.

        Args:
            timestep: The current timestep of the dynamic quantity
            position: The position of all particles at the new point in time
            orientation: The orientation (as a quaternion) of all particles

        Returns:
            Mapping of the names of each dynamic quantity to their values for each particle.

        Where a quantity can't be calculated, an array of nan values will be supplied
        instead, allowing for continued compatibility.

        """

        self.add(position, orientation)

        # Set default result
        dynamic_quantities = {key: np.nan for key in self._all_quantities}

        # Calculate all the simple dynamic quantities
        dynamic_quantities["time"] = self.compute_time_delta(timestep)
        dynamic_quantities["mean_displacement"] = np.linalg.norm(
            self.delta_translation, axis=1
        ).mean()
        dynamic_quantities["msd"] = self.compute_msd()
        dynamic_quantities["mfd"] = self.compute_mfd()
        dynamic_quantities["alpha"] = self.compute_alpha()

        # The scattering function takes too long to compute so is normally ignored.
        if scattering_function and self.wave_number is not None:
            dynamic_quantities["scattering_function"] = self.compute_isf()

        # The structural relaxation requires the distance value to be set
        if self.distance is not None:
            dynamic_quantities["com_struct"] = structural_relax(
                self.compute_displacement(), dist=self.distance
            )

        dynamic_quantities["mean_rotation"] = self.compute_mean_rotation()
        dynamic_quantities["msr"] = self.compute_msr()
        dynamic_quantities["rot1"] = self.compute_rotational_relax1()
        dynamic_quantities["rot2"] = self.compute_rotational_relax2()
        dynamic_quantities["alpha_rot"] = self.compute_alpha_rot()
        dynamic_quantities["gamma"] = self.compute_gamma()
        dynamic_quantities["overlap"] = mobile_overlap(
            self.compute_displacement(), self.compute_rotation()
        )

        # The structural relaxation of all atoms is the most complex.
        if (
            self.distance is not None
            and self.mol_vector is not None
            and self.motion.previous_orientation is not None
        ):
            dynamic_quantities["struct"] = self.compute_struct_relax()

        assert dynamic_quantities["time"] is not None

        return dynamic_quantities

    def __len__(self) -> int:
        return self.num_particles

    @property
    def distance(self) -> Optional[float]:
        if self.wave_number is None:
            return None
        return np.pi / (2 * self.wave_number)

    def get_molid(self):
        """Molecule ids of each of the values."""
        return np.arange(self.num_particles)


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
