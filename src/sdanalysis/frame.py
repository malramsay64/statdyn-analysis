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


class Frame(ABC):

    @property
    @abstractmethod
    def position(self) -> np.ndarray:
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


def lammpsFrame(Frame):

    def __init__(self, timestep: int, box, frame: Dict) -> None:
        self.frame = frame
        self.frame.box = np.array(box)

    @property
    def position(self):
        return np.array(
            [self.frame['x'], self.frame['y'], self.frame['z']], dtype=np.float32
        ).T

    @property
    def orientation(self) -> np.ndarray:
        return np.zeros((len(self), 4), dtype=np.float32)

    @property
    def timestep(self):
        return self.frame.timestep

    @property
    def box(self) -> np.ndarray:
        return self.frame.box.astype(np.float32)

    def __len__(self) -> int:
        return len(self.frame['x'])


def gsdFrame(Frame):

    def __init__(self, frame) -> None:
        self.frame = frame

    @property
    def position(self):
        return self.frame.particles.position

    @property
    def orientation(self):
        return self.frame.particles.orientation

    @property
    def timestep(self):
        return self.frame.configuration.step

    @property
    def box(self):
        return self.frame.configuration.box
