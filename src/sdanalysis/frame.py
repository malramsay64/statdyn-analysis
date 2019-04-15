#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Classes which hold frames."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import attr
import numpy as np
from freud.box import Box
from gsd.hoomd import Snapshot

from .util import create_freud_box


class Frame(ABC):
    @property
    @abstractmethod
    def position(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def image(self) -> Optional[np.ndarray]:
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

    @property
    @abstractmethod
    def dimensions(self) -> int:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def freud_box(self) -> Box:
        return create_freud_box(self.box, is_2D=self.dimensions == 2)


@attr.s(auto_attribs=True)
class LammpsFrame(Frame):
    frame: Dict

    @property
    def position(self) -> np.ndarray:
        return np.array(
            [self.frame["x"], self.frame["y"], self.frame["z"]], dtype=np.float32
        ).T

    @property
    def image(self) -> Optional[np.ndarray]:
        return None

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
        orient = np.zeros((len(self), 4), dtype=np.float32)
        orient[:, 0] = 1
        return orient

    @property
    def timestep(self) -> int:
        return self.frame["timestep"]

    @property
    def box(self) -> np.ndarray:
        box = np.zeros(6, dtype=np.float32)
        for index, value in enumerate(self.frame["box"]):
            if index > 5:
                break
            box[index] = value
        return box

    @property
    def dimensions(self) -> int:
        return 2

    def __len__(self) -> int:
        return len(self.frame["x"])


@attr.s(auto_attribs=True)
class HoomdFrame(Frame):
    frame: Snapshot
    _num_mols: int = attr.ib(init=False, default=0)

    def __attrs_post_init__(self):
        self._num_mols = self._get_num_bodies(self.frame)

    @classmethod
    def _get_num_bodies(cls, snapshot: Snapshot) -> int:
        num_particles = snapshot.particles.N
        try:
            num_mols = max(snapshot.particles.body) + 1
        except (AttributeError, ValueError, TypeError):
            num_mols = num_particles
        if num_mols > num_particles:
            num_mols = num_particles

        if num_mols > num_particles:
            raise ValueError(
                "Invalid Snapshot, there are more molecules than particles"
            )
        if num_mols > len(snapshot.particles.position):
            raise ValueError("There are more molecules than position datapoints.")

        return num_mols

    @property
    def num_mols(self):
        return self._num_mols

    @property
    def position(self) -> np.ndarray:
        return self.frame.particles.position[: self._num_mols]

    @property
    def image(self) -> Optional[np.ndarray]:
        return self.frame.particles.image

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
        # The snapshot from hoomd (vs gsd) has a different configuration
        if hasattr(self.frame, "box"):
            box = self.frame.box
            return np.array([box.Lx, box.Ly, box.Lz, box.xy, box.xz, box.yz])
        return self.frame.configuration.box

    @property
    def dimensions(self) -> int:
        if hasattr(self.frame, "box"):
            return self.frame.box.dimensions
        return self.frame.configuration.dimensions

    def __len__(self) -> int:
        return self._num_mols
