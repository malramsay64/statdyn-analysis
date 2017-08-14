#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Series of helper functions for the initialisation of parameters."""

import logging
from pathlib import Path

import hoomd
import hoomd.md as md

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def set_integrator(temperature: float,
                   tau: float=1.,
                   pressure: float=13.50,
                   tauP: float=1.,
                   step_size: float=0.005,
                   prime_interval: int=33533,
                   group: hoomd.group.group=None,
                   crystal: bool=False,
                   create: bool=True,
                   ) -> hoomd.md.integrate.npt:
    """Hoomd integrate method."""
    if group is None:
        group = hoomd.group.rigid_center()

    if create:
        md.update.enforce2d()
        md.integrate.mode_standard(step_size)
        if prime_interval:
            md.update.zero_momentum(period=prime_interval, phase=-1)

    if crystal:
        integrator = md.integrate.npt(
            group=group,
            kT=temperature,
            tau=tau,
            P=pressure,
            tauP=tauP,
            rescale_all=True,
            couple='none',
        )
    else:
        integrator = md.integrate.npt(
            group=group,
            kT=temperature,
            tau=tau,
            P=pressure,
            tauP=tauP,
        )
    return integrator


def dump_frame(outfile: Path,
               timestep: int=0,
               group: hoomd.group.group=None,
               ) -> None:
    """Dump frame to file."""
    if group is None:
        group = hoomd.group.rigid_center()
    hoomd.dump.gsd(
        str(outfile),
        period=None,
        time_step=timestep,
        group=group,
        overwrite=True,
        static=['topology', 'attribute']
    )


def set_dump(outfile: Path,
             dump_period: int=10000,
             ) -> None:
    """Initialise dumping configuration to a file."""
    hoomd.dump.gsd(
        str(outfile),
        period=dump_period,
        group=hoomd.group.all()
    )


def set_thermo(outfile: Path,
               thermo_period: int=10000
               ) -> None:
    """Set the thermodynamic quantities for a simulation."""
    default = ['N', 'volume', 'momentum', 'temperature', 'pressure',
               'potential_energy', 'kinetic_energy',
               'translational_kinetic_energy', 'rotational_kinetic_energy',
               'npt_thermostat_energy',
               'lx', 'ly', 'lz',
               'xy', 'xz', 'yz',
               ]
    rigid = ['temperature_rigid_center',
             'pressure_rigid_center',
             'potential_energy_rigid_center',
             'kinetic_energy_rigid_center',
             'translational_kinetic_energy_rigid_center',
             ]
    hoomd.analyze.log(
        str(outfile),
        quantities=default + rigid,
        period=thermo_period,
    )
