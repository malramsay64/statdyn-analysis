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


class lammpsFrame(Frame):

    def __init__(self, frame: Dict) -> None:
        self.frame = frame
        self.frame['box'] = np.array(self.frame['box'])

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
        return self.frame['timestep']

    @property
    def box(self) -> np.ndarray:
        return self.frame['box'].astype(np.float32)

    def __len__(self) -> int:
        return len(self.frame['x'])


class gsdFrame(Frame):

    def __init__(self, frame) -> None:
        self.frame = frame
        try:
            self._num_mols = min(
                max(self.frame.particles.body) + 1, len(self.frame.particles.body)
            )
        except AttributeError:
            self._num_mols = self.particles.N

    @property
    def position(self):
        return self.frame.particles.position[:self._num_mols]

    @property
    def orientation(self):
        return self.frame.particles.orientation[:self._num_mols]

    @property
    def timestep(self):
        return self.frame.configuration.step

    @property
    def box(self):
        return self.frame.configuration.box

    def __len__(self):
        return self._num_mols
