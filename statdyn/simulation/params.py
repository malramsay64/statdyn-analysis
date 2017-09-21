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
from typing import Any, Dict, Union

import hoomd

from ..crystals import Crystal
from ..molecules import Molecule


class SimulationParams(object):
    """Store the parameters of the simulation."""

    defaults = {
        'hoomd_args': '',
    }  # type: Dict[str, Any]

    def __init__(self, **kwargs) -> None:
        """Create SimulationParams instance."""
        self.parameters: Dict[str, Any] = deepcopy(self.defaults)
        self.parameters.update(**kwargs)

    def __getattr__(self, attr):
        return self.parameters.get(attr)

    def __setattr__(self, attr):
        return self.parameters.__setitem__(attr)

    def __delattr__(self, attr):
        return self.parameters.__delitem__(attr)

    @property
    def crystal(self) -> Crystal:
        """Return the crystal if it exists."""
        if self._crystal:
            return self._crystal
        raise ValueError('Crystal not found')

    @property
    def temperature(self) -> Union[float, hoomd.variant.linear_interp]:
        """Temperature of the system."""
        if self.init_temp:
            return hoomd.variant.linear_interp([
                (0, self._init_temp),
                (int(self.num_steps*0.75), self.parameters.get('temperature')),
                (self.num_steps, self.parameters.get('temperature')),
            ], zero='now')
        return self.parameters.get('temperature')

    @property
    def molecule(self) -> Molecule:
        """Return the appropriate molecule."""
        if self.crystal and not self.parameters.get('molecule'):
            return self.crystal.molecule
        return self.parameters.get('molecule')

    @property
    def group(self) -> hoomd.group.group:
        """Return the appropriate group."""
        if self.parameters.get('group'):
            return self.parameters.get('group')
        if self.molecule.num_particles == 1:
            return hoomd.group.all()
        return hoomd.group.rigid_center()

    @property
    def outdir(self) -> Path:
        """Ensure the output directory is a path."""
        if self.parameters.get('outdir'):
            return Path(self.parameters.get('outdir'))
        return Path.cwd()

    def set_group(self, group: hoomd.group.group) -> None:
        """Manually set integration group."""
        self.parameters['group'] = group

    def filename(self, prefix: str=None) -> str:
        """Use the simulation parameters to construct a filename."""
        return str(self.outfile_path / '-'.join(
            [str(value)
             for value in [prefix, self.molecule, self.pressure, self.temperature]
             if value]))
