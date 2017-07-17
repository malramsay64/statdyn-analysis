#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""A series of methods for the equilibration of configurations."""

from pathlib import Path

import hoomd

from ..molecule import Trimer
from .helper import dump_frame, set_integrator
from .initialise import initialise_snapshot


def create_interface(snapshot: hoomd.data.SnapshotParticleData,
                     melt_temp: float=2.50,
                     melt_steps: int=20000,
                     hoomd_args: str='',
                     tau: float=1.,
                     pressure: float=13.50,
                     tauP: float=1.,
                     step_size: float=0.005,
                     molecule=Trimer(),
                     outfile: Path=None,
                     ) -> hoomd.data.SnapshotParticleData:
    """Melt the middle segment of a crystal."""
    temp_context = hoomd.context.initialize(hoomd_args)
    sys = initialise_snapshot(
        snapshot=snapshot,
        context=temp_context,
        mol=Trimer()
    )
    with temp_context:
        set_integrator(temperature=melt_temp,
                       step_size=step_size,
                       prime_interval=307,
                       group=_interface_group(sys),
                       pressure=pressure,
                       tauP=tauP, tau=tau,
                       )
        hoomd.run(melt_steps)
        if outfile is not None:
            dump_frame(outfile, group=hoomd.group.all())
        return sys.take_snapshot(all=True)


def equil_interface(snapshot: hoomd.data.SnapshotParticleData,
                    equil_temp: float,
                    equil_steps: int,
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
        # Equilibrate crystal
        set_integrator(temperature=equil_temp,
                       step_size=step_size,
                       group=_interface_group(sys, stationary=True),
                       pressure=pressure,
                       tauP=tauP, tau=tau,
                       )
        hoomd.run(5000)
        # Equilibrate liquid
        set_integrator(temperature=equil_temp,
                       step_size=step_size,
                       group=_interface_group(sys, stationary=False),
                       pressure=pressure,
                       tauP=tauP, tau=tau,
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
        # Equilibrate crystal
        set_integrator(temperature=equil_temp,
                       step_size=step_size,
                       group=_interface_group(sys, stationary=True),
                       pressure=pressure,
                       tauP=tauP, tau=tau,
                       )
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
