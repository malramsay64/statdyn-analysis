#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Module for setting up and running a hoomd simulation."""

import logging
from pathlib import Path

import hoomd
import numpy as np

from . import initialise
from .. import molecule
from ..StepSize import GenerateStepSeries
from .helper import set_integrator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run_npt(snapshot: hoomd.data.SnapshotParticleData,
            context: hoomd.context.SimulationContext,
            output: Path,
            steps: int,
            temperature: float,
            pressure: float=13.50,
            dynamics: bool=True,
            max_initial: int=500,
            dump_period: int=10000,
            thermo_period: int=10000,
            mol: molecule.Molecule=molecule.Trimer(),
            ) -> None:
    """Initialise and run a hoomd npt simulation.

    Args:
        snapshot (class:`hoomd.data.snapshot`): Hoomd snapshot object
        temp (float): The temperature the simulation will run at
        steps (int): number of timesteps the simulation will run for

    Keyword Args:
        init_args (str): Args with which to initialise the hoomd context.
            Default: ''
        mol (class:`statdyn.Molecule`): Molecule to use in the simulation
            Default: class:`statdyn.Molecule.Trimer()`
        dt (float): size of each timestep in the simulation
            Default: 0.005
        tau (float): Restoring mass for the temperature integrator
            Default: 1.
        press (float): the pressure of the simulation
            Default: 13.5
        tauP (float): The restoring mass for the pressure integrator
            Default: 1.

    """
    with context:
        initialise.initialise_snapshot(
            snapshot=snapshot,
            context=context,
            mol=mol,
        )
        set_integrator(temperature=temperature)
        _set_thermo(temperature=temperature,
                    pressure=pressure,
                    output=output,
                    thermo_period=thermo_period,
                    )
        _set_dump(temperature=temperature,
                  pressure=pressure,
                  output=output,
                  dump_period=dump_period,
                  )
        if dynamics:
            iterator = GenerateStepSeries(steps, max_initial)
            prev_step = 0
            for curr_step in iterator:
                if curr_step == prev_step:
                    continue
                hoomd.run_upto(curr_step)
                _dump_frame(temperature=temperature,
                            pressure=pressure,
                            output=output,
                            timestep=curr_step)
                prev_step = curr_step
        else:
            hoomd.run(steps)
        _make_restart(temperature, pressure, output)


def _make_restart(temperature: float,
                  pressure: float,
                  output: Path
                  ) -> None:
    hoomd.dump.gsd(
        str(output / initialise.get_fname(temperature)),
        None,
        group=hoomd.group.all(),
        overwrite=True,
        )


def _set_thermo(temperature: float,
                pressure: float,
                output: Path,
                thermo_period: int=10000
                ) -> None:
    default = ['N', 'volume', 'momentum', 'temperature', 'pressure',
               'potential_energy', 'kinetic_energy',
               'translational_kinetic_energy', 'rotational_kinetic_energy',
               'npt_thermostat_energy']
    rigid = ['temperature_rigid_center',
             'pressure_rigid_center',
             'num_particles_rigid_center',
             'translational_ndof_rigid_center',
             'rotational_ndof_rigid_center',
             'potential_energy_rigid_center',
             'kinetic_energy_rigid_center',
             'translational_kinetic_energy_rigid_center',
             ]
    hoomd.analyze.log(
        str(output / f'thermo-{pressure:.2f}-{temperature:.2f}.log'),
        default + rigid,
        period=thermo_period,
    )


def _set_dump(temperature: float,
              pressure: float,
              output: Path,
              dump_period: int,
              ) -> None:
    hoomd.dump.gsd(
        str(output / f'dump-{pressure:.2f}-{temperature:.2f}.gsd'),
        period=dump_period,
        group=hoomd.group.all()
    )


def _dump_frame(temperature: float,
                pressure: float,
                output: Path,
                timestep: int,
                ) -> None:
    hoomd.dump.gsd(
        str(output / f'trajectory-{pressure:.2f}-{temperature:.2f}.gsd'),
        period=None,
        time_step=timestep,
        group=hoomd.group.rigid_center(),
        static=['topology', 'attribute']
    )


def read_snapshot(context: hoomd.context.SimulationContext,
                  fname: str,
                  rand: bool=False
                  ) -> hoomd.data.SnapshotParticleData:
    """Read a hoomd snapshot from a hoomd gsd file.

    Args:
    fname (string): Filename of GSD file to read in
    rand (bool): Whether to randomise the momenta of all the particles

    Returns:
    class:`hoomd.data.SnapshotParticleData`: Hoomd snapshot

    """
    with context:
        snapshot = hoomd.data.gsd_snapshot(fname)
        if rand:
            nbodies = snapshot.particles.body.max() + 1
            np.random.shuffle(snapshot.particles.velocity[:nbodies])
            np.random.shuffle(snapshot.particles.angmom[:nbodies])
            return snapshot
