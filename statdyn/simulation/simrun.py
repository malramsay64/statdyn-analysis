#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Module for setting up and running a hoomd simulation."""

from pathlib import Path

import hoomd
import hoomd.md as md
import numpy as np
import pandas

from . import initialise
from ..StepSize import GenerateStepSeries


def set_defaults(kwargs):
    """Set the default values for parameters."""
    kwargs.setdefault('output', Path('.'))
    kwargs.setdefault('init_args', '')
    kwargs.setdefault('tau', 1.)
    kwargs.setdefault('press', 13.5)
    kwargs.setdefault('tauP', 1.)
    kwargs.setdefault('dt', 0.005)
    kwargs.setdefault('thermo', True)
    kwargs.setdefault('thermo_dir', Path(kwargs.get('output', '.')))
    kwargs.setdefault('thermo_period', 10000)
    kwargs.setdefault('dump', True)
    kwargs.setdefault('dump_dir', Path(kwargs.get('output', '.')))
    kwargs.setdefault('dump_period', 50000)
    kwargs.setdefault('restart', True)
    kwargs.setdefault('dyn_many', True)
    kwargs.setdefault('max_gen', 500)


def run_npt(snapshot: hoomd.data.SnapshotParticleData,
            temp: float,
            steps: int,
            **kwargs) -> Path:
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
    set_defaults(kwargs)
    context = hoomd.context.initialize(kwargs.get('init_args'))
    kwargs['context'] = context
    kwargs['temp'] = temp
    with context:
        initialise.init_from_snapshot(snapshot, **kwargs)
        _set_integrator(kwargs)
        _set_thermo(kwargs)
        _set_dump(kwargs)
        if kwargs.get('dyn_many'):
            iterator = GenerateStepSeries(steps, kwargs.get('max_gen'))
        else:
            iterator = GenerateStepSeries(steps, max_gen=1)
        prev_step = 0
        for curr_step in iterator:
            if curr_step == prev_step:
                continue
            hoomd.run_upto(curr_step)
            _dump_frame(kwargs, curr_step)
            prev_step = curr_step
        _make_restart(kwargs)


def _make_restart(kwargs):
    if kwargs.get('restart'):
        hoomd.dump.gsd(
            str(kwargs.get('output') /
                initialise.get_fname(kwargs.get('temp'))),
            None,
            group=hoomd.group.all(),
            overwrite=True,
        )


def _set_integrator(kwargs):
    md.update.enforce2d()
    prime_interval = 33533
    md.update.zero_momentum(period=prime_interval)
    md.integrate.mode_standard(kwargs.get('dt'))
    md.integrate.npt(
        group=hoomd.group.rigid_center(),
        kT=kwargs.get('temp'),
        tau=kwargs.get('tau'),
        P=kwargs.get('press'),
        tauP=kwargs.get('tauP')
    )


def _set_thermo(kwargs):
    if kwargs.get('thermo'):
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
            str(kwargs.get('thermo_dir') /
                'thermo-{press:.2f}-{temp:.2f}.log'.format(
                press=kwargs.get('press'), temp=kwargs.get('temp'))),
            default + rigid,
            period=kwargs.get('thermo_period'),
        )


def _set_dump(kwargs):
    if kwargs.get('dump'):
        hoomd.dump.gsd(
            str(kwargs.get('dump_dir') /
                'dump-{press:.2f}-{temp:.2f}.gsd'.format(
                press=kwargs.get('press'), temp=kwargs.get('temp'))
                ),
            period=kwargs.get('dump_period'),
            group=hoomd.group.all()
        )


def _dump_frame(kwargs, timestep):
    hoomd.dump.gsd(
        str(kwargs.get('output') /
            'trajectory-{press:.2f}-{temp:.2f}.gsd'.format(
            press=kwargs.get('press'), temp=kwargs.get('temp'))
            ),
        period=None,
        time_step=timestep,
        group=hoomd.group.rigid_center(),
        static=['topology', 'attribute']
    )


def read_snapshot(fname: str,
                  rand: bool=False) -> hoomd.data.SnapshotParticleData:
    """Read a hoomd snapshot from a hoomd gsd file.

    Args:
        fname (string): Filename of GSD file to read in
        rand (bool): Whether to randomise the momenta of all the particles

    Returns:
        class:`hoomd.data.SnapshotParticleData`: Hoomd snapshot

    """
    with hoomd.context.initialize(''):
        snapshot = hoomd.data.gsd_snapshot(fname)
        if rand:
            nbodies = snapshot.particles.body.max() + 1
            np.random.shuffle(snapshot.particles.velocity[:nbodies])
            np.random.shuffle(snapshot.particles.angmom[:nbodies])
            return snapshot


def iterate_random(directory: Path,
                   temp: float,
                   steps: int,
                   iterations: int=2,
                   output: Path=Path('.'),
                   **kwargs) -> None:
    """Iterate over a configuration initialised with randomised momenta.

    This function will take a single configuration and then run `iterations`
    iterations initialising each with a different random momenta for both
    translations and rotations.

    Args:
        directory (str): dir of input files
        temp (float): temp of simulation
        steps (int): number of steps to run simulation
        iterations(int): number of iterations of length steps to run
        output (str): directory to output data

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
    init_file = directory / "Trimer-{press:.2f}-{temp:.2f}.gsd".format(
        press=kwargs.get('press', 13.5),
        temp=temp
    )
    for iteration in range(iterations):
        dynamics = run_npt(
            read_snapshot(str(init_file), rand=True),
            temp,
            steps,
            output=output,
            **kwargs
        )
        with pandas.HDFStore(dynamics) as store:
            store.get_node('dynamics')._f_rename('dyn{i}'.format(i=iteration))
