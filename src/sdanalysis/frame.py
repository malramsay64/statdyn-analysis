#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Classes which hold frames."""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from dataclasses import dataclass
from gsd.hoomd import Snapshot


class Frame(ABC):
    @property
    @abstractmethod
    def position(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def x_position(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def y_position(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def z_position(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def orientation(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def box(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def timestep(self) -> int:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


@dataclass
class lammpsFrame(Frame):
    frame: Dict

    def __post_init__(self):
        self.frame["box"] = np.array(self.frame["box"])

    @property
    def position(self) -> np.ndarray:
        return np.array(
            [self.frame["x"], self.frame["y"], self.frame["z"]], dtype=np.float32
        ).T

    @property
    def x_position(self) -> np.ndarray:
        return self.frame["x"].astype(np.float32)

    @property
    def y_position(self) -> np.ndarray:
        return self.frame["y"].astype(np.float32)

    @property
    def z_position(self) -> np.ndarray:
        return self.frame["z"].astype(np.float32)

    @property
    def orientation(self) -> np.ndarray:
        return np.zeros((len(self), 4), dtype=np.float32)

    @property
    def timestep(self) -> int:
        return self.frame["timestep"]

    @property
    def box(self) -> np.ndarray:
        return self.frame["box"].astype(np.float32)

    def __len__(self) -> int:
        return len(self.frame["x"])


@dataclass
class gsdFrame(Frame):
    frame: Snapshot
    _num_mols: int = 0

    def __post_init__(self):
        num_particles = snapshot.particles.N
        try:
            self._num_mols = min(
                max(self.frame.particles.body) + 1, len(self.frame.particles.body)
            )
        except AttributeError:
            self._num_mols = self.frame.particles.N

    @property
    def position(self) -> np.ndarray:
        return self.frame.particles.position[: self._num_mols]

    @property
    def x_position(self) -> np.ndarray:
        return self.frame.particles.position[: self._num_mols, 0]

    @property
    def y_position(self) -> np.ndarray:
        return self.frame.particles.position[: self._num_mols, 1]

    @property
    def z_position(self) -> np.ndarray:
        return self.frame.particles.position[: self._num_mols, 2]

    @property
    def orientation(self) -> np.ndarray:
        return self.frame.particles.orientation[: self._num_mols]

    @property
    def timestep(self) -> int:
        return self.frame.configuration.step

    @property
    def box(self) -> np.ndarray:
        return self.frame.configuration.box

    def __len__(self) -> int:
        return self._num_mols
