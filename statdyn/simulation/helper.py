#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Series of helper functions for the initialisation of parameters."""

import logging
from functools import partial
from pathlib import Path
from typing import List, Union

import hoomd
import hoomd.md as md

from ..crystals import Crystal
from ..molecules import Molecule

logger = logging.getLogger(__name__)


class SimulationParams(object):
    """Store the parameters of the simulation."""

    def __init__(self,
                 temperature: float,
                 num_steps: int,
                 molecule: Molecule=None,
                 tau: float=1.,
                 pressure: float=13.50,
                 tauP: float=1.,
                 step_size: float=0.005,
                 init_temp: float=None,
                 hoomd_args: str='',
                 outfile_path: Path=Path.cwd(),
                 crystal: Crystal=None,
                 ) -> None:
        """Create SimulationParams instance."""
        if molecule is None and crystal is None:
            raise ValueError('Both molecule and crystal are None, one of them needs to be defined.')
        self._target_temp = temperature
        self.tau = tau
        self.pressure = pressure
        self.tauP = tauP
        self.step_size = step_size
        self._molecule = molecule
        self.num_steps = num_steps
        self._init_temp = init_temp
        self.outfile_path = outfile_path
        self.hoomd_args = hoomd_args
        self._crystal = crystal
        self._group: hoomd.group.group = None

    @property
    def crystal(self) -> Crystal:
        """Return the crystal if it exists."""
        if self._crystal:
            return self._crystal
        raise ValueError('Crystal not found')

    @property
    def temperature(self) -> Union[float, hoomd.variant.linear_interp]:
        """Temperature of the system."""
        if self._init_temp:
            return hoomd.variant.linear_interp([
                (0, self._init_temp),
                (int(self.num_steps*0.75), self._target_temp),
                (self.num_steps, self._target_temp),
            ], zero='now')
        return self._target_temp

    @property
    def molecule(self) -> Molecule:
        """Return the appropriate molecule."""
        if self.crystal:
            return self.crystal.molecule
        return self._molecule

    @property
    def group(self) -> hoomd.group.group:
        """Return the appropriate group."""
        if self._group:
            return self._group
        if self.molecule.num_particles == 1:
            return hoomd.group.all()
        return hoomd.group.rigid_center()

    def set_group(self, group: hoomd.group.group) -> None:
        """Manually set integration group."""
        self._group = group

    def filename(self, prefix: str=None) -> str:
        """Use the simulation parameters to construct a filename."""
        return str(self.outfile_path / '-'.join(
            [str(value)
             for value in [prefix, self.molecule, self.pressure, self.temperature]
             if value]))


def set_integrator(sim_params: SimulationParams,
                   prime_interval: int=33533,
                   crystal: bool=False,
                   create: bool=True,
                   ) -> hoomd.md.integrate.npt:
    """Hoomd integrate method."""
    if create:
        md.update.enforce2d()
        md.integrate.mode_standard(sim_params.step_size)
        if prime_interval:
            md.update.zero_momentum(period=prime_interval, phase=-1)

    if crystal:
        integrator = md.integrate.npt(
            group=sim_params.group,
            kT=sim_params.temperature,
            tau=sim_params.tau,
            P=sim_params.pressure,
            tauP=sim_params.tauP,
            rescale_all=True,
            couple='none',
        )
    else:
        integrator = md.integrate.npt(
            group=sim_params.group,
            kT=sim_params.temperature,
            tau=sim_params.tau,
            P=sim_params.pressure,
            tauP=sim_params.tauP,
        )
    return integrator


def set_dump(outfile: str,
             dump_period: int=10000,
             timestep: int=0,
             group: hoomd.group.group=None,
             ) -> hoomd.dump.gsd:
    """Initialise dumping configuration to a file."""
    if group is None:
        group = hoomd.group.rigid_center()
    # TODO on update to hoomd 2.2.0 change static to dynamic
    return hoomd.dump.gsd(
        outfile+'.gsd',
        time_step=timestep,
        period=dump_period,
        group=group,
        overwrite=True,
        static=['topology', 'attribute', 'momentum']
    )


dump_frame = partial(set_dump, dump_period=None)


def set_thermo(outfile: str,
               thermo_period: int=10000,
               rigid=True,
               ) -> None:
    """Set the thermodynamic quantities for a simulation."""
    default = ['N', 'volume', 'momentum', 'temperature', 'pressure',
               'potential_energy', 'kinetic_energy',
               'translational_kinetic_energy', 'rotational_kinetic_energy',
               'npt_thermostat_energy',
               'lx', 'ly', 'lz',
               'xy', 'xz', 'yz',
               ]
    rigid_thermo: List[str] = []
    if rigid:
        rigid_thermo = ['temperature_rigid_center',
                        'pressure_rigid_center',
                        'potential_energy_rigid_center',
                        'kinetic_energy_rigid_center',
                        'translational_kinetic_energy_rigid_center',
                        'translational_ndof_rigid_center',
                        'rotational_ndof_rigid_center',
                        ]
    # TODO Set logger to hdf5 file on update to hoomd 2.2.0
    hoomd.analyze.log(
        outfile+'.log',
        quantities=default + rigid_thermo,
        period=thermo_period,
    )
