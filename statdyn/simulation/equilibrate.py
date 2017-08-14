#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""A series of methods for the equilibration of configurations."""

import logging
from pathlib import Path

import hoomd
import hoomd.md

from ..molecule import Trimer
from .helper import dump_frame, set_dump, set_integrator, set_thermo
from .initialise import initialise_snapshot, make_orthorhombic

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def equil_crystal(snapshot: hoomd.data.SnapshotParticleData,
                  equil_temp: float=0.4,
                  equil_steps: int=5000,
                  hoomd_args: str='',
                  tau: float=1.,
                  pressure: float=13.50,
                  tauP: float=1.,
                  step_size: float=0.005,
                  interface: bool=False,
                  molecule=Trimer(),
                  outfile: Path=None,
                  ) -> hoomd.data.SnapshotParticleData:
    """Equilbrate crystal."""
    temp_context = hoomd.context.initialize(hoomd_args)
    sys = initialise_snapshot(
        snapshot=snapshot,
        context=temp_context,
        mol=molecule
    )

    if interface:
        group = _interface_group(sys)
    else:
        group = None

    with temp_context:
        init_temp = 0.4
        temperature = hoomd.variant.linear_interp([
            (0, init_temp),
            (int(equil_steps*0.75), equil_temp),
            (equil_steps, equil_temp),
        ], zero='now')
        set_integrator(
            temperature=temperature,
            step_size=step_size,
            prime_interval=307,
            group=group,
            pressure=pressure,
            crystal=True,
            tauP=tauP, tau=tau,
        )

        if outfile is not None:
            set_dump(outfile.parent / ('dump-' + outfile.name))

        logger.debug(f'Running crystal equilibration for {equil_steps} steps.')
        set_thermo(Path('equil.log'), thermo_period=1)
        hoomd.run(equil_steps)
        logger.debug(f'Crystal equilibration completed')

        return sys.take_snapshot()
        if outfile is not None:
            dump_frame(outfile, group=hoomd.group.all())

        return make_orthorhombic(equil_snap)
        return sys.take_snapshot()


def equil_interface(snapshot: hoomd.data.SnapshotParticleData,
                    equil_temp: float,
                    equil_steps: int,
                    init_temp: float=4.00,
                    hoomd_args: str='',
                    tau: float=1.,
                    pressure: float=13.50,
                    tauP: float=1.,
                    step_size: float=0.005,
                    molecule=Trimer(),
                    outfile: Path=None,
                    ) -> hoomd.data.SnapshotParticleData:
    """Equilbrate an interface at the desired temperature.

    This is first done by equilibrating the crystal phase, which once completed
    the liquid phase is equilibrated.
    """
    temp_context = hoomd.context.initialize(hoomd_args)
    sys = initialise_snapshot(
        snapshot=snapshot,
        context=temp_context,
        mol=molecule,
    )
    with temp_context:
        # Equilibrate liquid
        if init_temp:
            temperature = hoomd.variant.linear_interp([
                (0, init_temp),
                (equil_steps/2, equil_temp),
                (equil_steps, equil_temp)
            ])
        else:
            temperature = equil_temp
        set_integrator(temperature=temperature,
                       step_size=step_size,
                       group=_interface_group(sys, stationary=False),
                       pressure=pressure,
                       tauP=tauP, tau=tau,
                       crystal=True,
                       create=False,
                       )
        hoomd.run(equil_steps)
        if outfile is not None:
            dump_frame(outfile, group=hoomd.group.all())
        return sys.take_snapshot(all=True)


def equil_liquid(snapshot: hoomd.data.SnapshotParticleData,
                 equil_temp: float,
                 equil_steps: int,
                 hoomd_args: str='',
                 tau: float=1.,
                 pressure: float=13.50,
                 tauP: float=1.,
                 step_size: float=0.005,
                 molecule=Trimer(),
                 init_temp: float=None,
                 outfile: Path=None,
                 ) -> hoomd.data.SnapshotParticleData:
    """Equilibrate a liquid configuration."""
    temp_context = hoomd.context.initialize(hoomd_args)
    sys = initialise_snapshot(
        snapshot=snapshot,
        context=temp_context,
        mol=molecule
    )
    with temp_context:
        if init_temp:
            temperature = hoomd.variant.linear_interp([
                (0, init_temp),
                (equil_steps/2, equil_temp),
                (equil_steps, equil_temp)
            ])
        else:
            temperature = equil_temp

        set_integrator(
            temperature=temperature,
            step_size=step_size,
            group=None,
            pressure=pressure,
            tauP=tauP, tau=tau,
        )
        if outfile is not None:
            set_thermo(outfile.parent / 'thermo-{temperature:.2f}.log')
        hoomd.run(equil_steps)
        if outfile is not None:
            dump_frame(outfile, group=hoomd.group.all())
        return sys.take_snapshot(all=True)


def _interface_group(sys: hoomd.data.system_data,
                     stationary: bool=False):
    stationary_group = hoomd.group.cuboid(name='stationary',
                                          xmin=-sys.box.Lx/4,
                                          xmax=sys.box.Lx/4)
    if stationary:
        return hoomd.group.intersection(
            'rigid_stationary',
            stationary_group,
            hoomd.group.rigid_center()
        )
    return hoomd.group.intersection(
        'rigid_mobile',
        hoomd.group.difference('mobile', hoomd.group.all(), stationary_group),
        hoomd.group.rigid_center(),
    )
