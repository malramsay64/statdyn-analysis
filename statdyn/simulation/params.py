#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Parameters for passing between functions."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import hoomd

from ..crystals import Crystal
from ..molecules import Molecule, Trimer


class SimulationParams(object):
    """Store the parameters of the simulation."""

    defaults = {
        'hoomd_args': '',
        'step_size': 0.005,
        'temperature': 0.4,
        'tau': 1.0,
        'pressure': 13.5,
        'tauP': 1.0,
        'cell_dimensions': (30, 42),
        'outfile_path': Path.cwd(),
        'max_gen': 500,
        'gen_steps': 20_000,
        'output_interval': 10_000,
    }  # type: Dict[str, Any]

    def __init__(self, **kwargs) -> None:
        """Create SimulationParams instance."""
        self.parameters: Dict[str, Any] = deepcopy(self.defaults)
        self.parameters.update(kwargs)

    # I am using getattr over getattribute becuase of the lower search priority
    # of getattr. This makes it a fallback, rather than the primary location
    # for looking up attributes.
    def __getattr__(self, key):
        try:
            return self.parameters.__getitem__(key)
        except KeyError:
            raise AttributeError

    def __setattr__(self, key, value):
        # setattr has a higher search priority than other functions, custom
        # setters need to be added to the list below
        if key in ['parameters']:
            super().__setattr__(key, value)
        else:
            self.parameters.__setitem__(key, value)

    def __delattr__(self, attr):
        return self.parameters.__delitem__(attr)

    @property
    def temperature(self) -> Union[float, hoomd.variant.linear_interp]:
        """Temperature of the system."""
        try:
            return hoomd.variant.linear_interp([
                (0, self.init_temp),
                (int(self.num_steps*0.75), self.parameters.get('temperature', self.init_temp)),
                (self.num_steps, self.parameters.get('temperature', self.init_temp)),
            ], zero='now')
        except AttributeError:
            return self.parameters.get('temperature')

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.parameters['temperature'] = value

    @property
    def molecule(self) -> Molecule:
        """Return the appropriate molecule.

        Where there is no custom molecule defined then we return the molecule of
        the crystal.

        """
        if self.parameters.get('crystal') is not None and self.parameters.get('molecule') is None:
            return self.crystal.molecule
        return self.parameters.get('molecule', Trimer())

    @property
    def cell_dimensions(self) -> Tuple[int, int]:
        try:
            self.crystal
            return self.parameters.get('cell_dimensions')
        except AttributeError:
            raise AttributeError

    @property
    def group(self) -> hoomd.group.group:
        """Return the appropriate group."""
        if self.parameters.get('group'):
            return self.parameters.get('group')
        if self.molecule.num_particles == 1:
            return hoomd.group.all()
        return hoomd.group.rigid_center()

    @property
    def outfile_path(self) -> Path:
        """Ensure the output directory is a path."""
        if self.parameters.get('outfile_path'):
            return Path(self.parameters.get('outfile_path'))
        return Path.cwd()

    @property
    def outfile(self) -> Path:
        """Ensure the output directory is a path."""
        return Path(self.parameters.get('outfile'))

    def filename(self, prefix: str=None) -> str:
        """Use the simulation parameters to construct a filename."""
        return str(self.outfile_path / '-'.join(
            [str(value)
             for value in [prefix, self.molecule, self.pressure, self.temperature]
             if value]))
