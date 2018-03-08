#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Classes which hold frames."""

from typing import Dict
from abc import ABC, abstractmethod
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

