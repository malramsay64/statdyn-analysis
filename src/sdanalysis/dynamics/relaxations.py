#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Calculate molecular relaxation quantities for all molecules in a simulation.

The molecular relaxation quantities compute the relaxation time for each molecule independently,
rather than the traditional approach which is averaged over all particles. 
Giving each particle a relaxation time provides additional tools for understanding motion,
including the ability to get spatial maps of relaxation.

"""
import logging
from enum import Enum, auto
from typing import Dict, List, Optional, Union

import numpy as np
import pandas

from ..frame import Frame
from ..molecules import Molecule
from ..util import create_freud_box
from ._util import TrackedMotion, calculate_wave_number

YamlValue = Union[str, float, int]

np.seterr(divide="raise", invalid="raise", over="raise")
logger = logging.getLogger(__name__)


class RelaxationType(Enum):
    rotation = auto()
    translation = auto()


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

        box = create_freud_box(box, is_2D=is2D)
        self.motion = TrackedMotion(box, position, orientation)
        self._num_elements = position.shape[0]

        if wave_number is None:
            wave_number = calculate_wave_number(box, position)

        self.wave_number = wave_number

        # set defualt values for mol_relax
        self.set_mol_relax(
            [
                {"name": "tau_D", "threshold": 3 * self.distance},
                {"name": "tau_F", "threshold": self.distance},
                {
                    "name": "tau_L",
                    "threshold": self.distance,
                    "last_passage": True,
                    "last_passage_cutoff": 3 * self.distance,
                },
                {"name": "tau_T2", "threshold": np.pi / 2, "type": "rotation"},
                {"name": "tau_T3", "threshold": np.pi / 3, "type": "rotation"},
                {"name": "tau_T4", "threshold": np.pi / 4, "type": "rotation"},
            ]
        )

    @property
    def distance(self):
        return np.pi / (2 * self.wave_number)

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
            last_passage_cutoff = float(
                item.get("last_passage_cutoff", 3.0 * threshold)
            )
            relaxation_type = str(item.get("type", "translation"))

            self.mol_relax[index] = create_mol_relaxations(
                self._num_elements,
                threshold=threshold,
                last_passage=last_passage,
                last_passage_cutoff=last_passage_cutoff,
                relaxation_type=relaxation_type,
            )

    def get_timediff(self, timestep: int):
        return timestep - self.init_time

    def add(self, timestep: int, position: np.ndarray, orientation: np.ndarray) -> None:
        """Update the state of the relaxation calculations by adding a Frame.

        This updates the motion of the particles, comparing the positions and
        orientations of the current frame with the previous frame, adding the difference
        to the total displacement. This approach allows for tracking particles over
        periodic boundaries, or through larger rotations assuming that there are
        sufficient frames to capture the information. Each single displacement obeys the
        minimum image convention, so for large time intervals it is still possible to
        have missing information.

        Args:
            timestep: The timestep of the frame being added
            position: The new position of each particle in the simulation
            orientation: The updated orientation of each particle, represented as a quaternion.

        """
        self.motion.add(position, orientation)
        displacement = np.linalg.norm(self.motion.delta_translation, axis=1)
        rotation = np.linalg.norm(self.motion.delta_rotation, axis=1)

        for _, func in self.mol_relax.items():
            if func.relaxation_type is RelaxationType.rotation:
                func.add(self.get_timediff(timestep), rotation)
            elif func.relaxation_type is RelaxationType.translation:
                func.add(self.get_timediff(timestep), displacement)
            else:
                raise RuntimeError("Invalid relaxation type")

    def add_frame(self, frame: Frame):
        """Update the state of the relaxation calculations by adding a Frame.

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
        self.add(frame.timestep, frame.position, frame.orientation)

    def summary(self) -> pandas.DataFrame:
        return pandas.DataFrame(
            {key: func.get_status() for key, func in self.mol_relax.items()}
        )


class MolecularRelaxation:
    """Compute the relaxation of each molecule."""

    _max_value = 2 ** 32 - 1
    relaxation_type = RelaxationType.translation

    def __init__(
        self, num_elements: int, threshold: float, relaxation_type: Optional[str] = None
    ) -> None:
        self.num_elements = num_elements
        self.threshold = threshold
        self._max_value = 2 ** 32 - 1
        self._status = np.full(self.num_elements, self._max_value, dtype=int)

        if relaxation_type is not None:
            logger.debug(
                "relaxation_type: %s, type(relaxation_type): %s",
                relaxation_type,
                type(relaxation_type),
            )
            self.relaxation_type = RelaxationType[relaxation_type]

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
    _is_irreversible = 2

    def __init__(
        self,
        num_elements: int,
        threshold: float,
        irreversibility: float = 1.0,
        relaxation_type: Optional[str] = None,
    ) -> None:
        super().__init__(num_elements, threshold, relaxation_type)
        self._state = np.zeros(self.num_elements, dtype=np.uint8)
        self._irreversibility = irreversibility

    def add(self, timediff: int, distance: np.ndarray) -> None:
        if distance.shape != self._status.shape:
            raise RuntimeError(
                "Current state and initial state have different shapes. "
                "current: {distance.shape}, initial: {self._status.shape}"
            )

        with np.errstate(invalid="ignore"):
            # The state can have one of three values,
            #   - 0 => No relaxation has taken place, or particle has moved back within
            #          the threshold distance
            #   - 1 => The distance has passed the threshold, however not the
            #          irreversibility distance
            #   - 2 => The molecule has passed the irreversibility distance
            #
            #  The following is a sample linear algorithm for the process
            #
            #  for state, dist, status in zip(self._state, distance, self._status):
            #      if state == 2:
            #          # The threshold has been reached so there is now nothing to do
            #          continue
            #      elif state == 1:
            #          # Need to check whether the relaxation has crossed a threshold
            #          if dist > self._irreversibility:
            #              state = 2
            #          elif dist < self.threshold:
            #              state = 0
            #          else:
            #              continue
            #      elif state == 0:
            #          if dist > self.threshold:
            #              status = timediff
            #              state = 1
            #          else:
            #              continue
            #      else:
            #          RuntimeError("Invalid State")

            # These are the molecules which have moved from state 0 to state 1
            # This movement is accompanied by updating the time this motion occurred
            passed_threshold = (distance > self.threshold) & (self._state == 0)
            self._state[passed_threshold] = 1
            self._status[passed_threshold] = timediff

            # These are the molecules which have moved from state 1 to state 0
            below_threshold = (distance < self.threshold) & (self._state == 1)
            self._state[below_threshold] = 0

            # These are the molecules which have moved from state 1 to state 2
            # This is at the end so molecules can go from state 0 to 2 in one jump.
            irreversible = (distance > self._irreversibility) & (self._state == 1)
            self._state[irreversible] = 2

    def get_status(self):
        status = np.copy(self._status)
        status[self._state != 2] = self._max_value
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
    relaxation_type: Optional[str] = None,
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

        return LastMolecularRelaxation(
            num_elements, threshold, last_passage_cutoff, relaxation_type
        )

    return MolecularRelaxation(num_elements, threshold, relaxation_type)
