#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Parameters for passing between functions."""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import attr

from .molecules import Molecule, Trimer

logger = logging.getLogger(__name__)


def _to_path(value: Optional[Path]) -> Optional[Path]:
    if value is not None:
        return Path(value)
    return value


@attr.s(auto_attribs=True)
class SimulationParams:
    """Store the parameters of the simulation."""

    # Thermodynamic Params
    temperature: float = 0.4
    pressure: float = 13.5

    # Molecule params
    molecule: Molecule = Trimer()
    moment_inertia_scale: Optional[float] = None
    harmonic_force: Optional[float] = None
    wave_number: Optional[float] = None

    # Crystal Params
    space_group: Optional[str] = None

    # Step Params
    num_steps: Optional[int] = None
    linear_steps: int = attr.ib(default=100, repr=False)
    max_gen: int = attr.ib(default=500, repr=False)
    gen_steps: int = attr.ib(default=20000, repr=False)
    output_interval: int = attr.ib(default=10000, repr=False)

    # File Params
    _infile: Optional[Path] = attr.ib(default=None, converter=_to_path, repr=False)
    _outfile: Optional[Path] = attr.ib(default=None, converter=_to_path, repr=False)
    _output: Optional[Path] = attr.ib(default=None, converter=_to_path, repr=False)

    @property
    def infile(self) -> Optional[Path]:
        return self._infile

    @infile.setter
    def infile(self, value: Path) -> None:
        # Ensure value is a Path
        self._infile = Path(value)

    @property
    def outfile(self) -> Optional[Path]:
        return self._outfile

    @outfile.setter
    def outfile(self, value: Optional[Path]) -> None:
        # Ensure value is a Path
        if value is not None:
            self._outfile = Path(value)

    @property
    def output(self) -> Path:
        if self._output is None:
            return Path.cwd()
        return self._output

    @output.setter
    def output(self, value: Optional[Path]) -> None:
        # Ensure value is a Path
        if value is not None:
            self._output = Path(value)

    def filename(self, prefix: str = None) -> Path:
        """Use the simulation parameters to construct a filename."""
        base_string = "{molecule}-P{pressure:.2f}-T{temperature:.2f}"
        if prefix is not None:
            base_string = "{prefix}-" + base_string
        if self.moment_inertia_scale is not None:
            base_string += "-I{mom_inertia:.2f}"
        if self.harmonic_force is not None:
            base_string += "-K{harmonic_force:.2f}"

        if self.space_group is not None:
            base_string += "-{space_group}"

        logger.debug("filename base string: %s", base_string)
        logger.debug("Temperature: %.2f", self.temperature)

        # Default extension, required as with_suffix replaces existsing extension
        # which is mistaken for the final decimal points.
        base_string += ".gsd"

        fname = base_string.format(
            prefix=prefix,
            molecule=self.molecule,
            pressure=self.pressure,
            temperature=self.temperature,
            mom_inertia=self.moment_inertia_scale,
            space_group=self.space_group,
            harmonic_force=self.harmonic_force,
        )
        return self.output / fname

    @contextmanager
    def temp_context(self, **kwargs):
        old_params = {
            key: val
            for key, val in self.__dict__.items()
            if not isinstance(val, property)
        }
        for key, value in kwargs.items():
            setattr(self, key, value)
        yield self
        self.__dict__.update(old_params)
