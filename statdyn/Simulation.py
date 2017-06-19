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
from statdyn import TimeDep, initialise
from statdyn.StepSize import GenerateStepSeries


def set_defaults(kwargs):
    """Set the default values for parameters."""
    kwargs.setdefault('init_args', '')
    kwargs.setdefault('tau', 1.)
    kwargs.setdefault('press', 13.5)
    kwargs.setdefault('tauP', 1.)
    kwargs.setdefault('dt', 0.005)
    kwargs.setdefault('thermo', True)
    kwargs.setdefault('thermo_dir', '.')
    kwargs.setdefault('thermo_period', 10000)
    kwargs.setdefault('dump', True)
    kwargs.setdefault('dump_dir', '.')
    kwargs.setdefault('dump_period', 50000)
    kwargs.setdefault('restart', True)
    kwargs.setdefault('dyn_many', True)


def run_npt(snapshot: hoomd.data.SnapshotParticleData,
            temp: float,
            steps: int,
            **kwargs) -> pandas.DataFrame:
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
        system = initialise.init_from_snapshot(snapshot, **kwargs)
        _set_integrator(kwargs)
        _set_thermo(kwargs)
        _set_dump(kwargs)
        dynamics = TimeDep.TimeDepMany()
        dynamics.append(system.take_snapshot(all=True), 0, 0)
        if kwargs.get('dyn_many'):
            iterator = GenerateStepSeries(steps)
        else:
            iterator = GenerateStepSeries(steps, max_gen=1)
        for curr_step in iterator:
            hoomd.run_upto(curr_step)
            dynamics.append(system.take_snapshot(all=True),
                            iterator.get_index(), curr_step)
        _make_restart(kwargs)
    return dynamics.get_all_data()


def _make_restart(kwargs):
    if kwargs.get('restart'):
        hoomd.dump.gsd(
            initialise.get_fname(kwargs.get('temp')),
            None,
            group=hoomd.group.all(),
            overwrite=True,
        )


def _set_integrator(kwargs):
    md.update.enforce2d()
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
        hoomd.analyze.log(
            kwargs.get('thermo_dir') + '/' +
            'thermo-{press:.2f}-{temp:.2f}.log'.format(
                press=kwargs.get('press'), temp=kwargs.get('temp')),
            ['volume', 'potential_energy', 'kinetic_energy',
             'rotational_kinetic_energy', 'temperature', 'pressure'],
            period=kwargs.get('thermo_period'),
        )


def _set_dump(kwargs):
    if kwargs.get('dump'):
        hoomd.dump.gsd(
            kwargs.get('dump_dir') + '/' +
            'dump-{press:.2f}-{temp:.2f}.gsd'.format(
                press=kwargs.get('press'), temp=kwargs.get('temp')),
            period=kwargs.get('dump_period'),
            group=hoomd.group.all()
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


def iterate_random(directory: str,
                   temp: float,
                   steps: int,
                   iterations: int=2,
                   output: str='.',
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
    init_file = Path(directory) / "Trimer-{press:.2f}-{temp:.2f}.gsd".format(
        press=kwargs.get('press', 13.5),
        temp=temp
    )
    for iteration in range(iterations):
        dynamics = run_npt(
            read_snapshot(str(init_file), rand=True),
            temp,
            steps,
            **kwargs
        )
        outfile = Path(output) / (init_file.stem + '.hdf5')
        with pandas.HDFStore(outfile) as store:
            store['dyn{i}'.format(i=iteration)] = dynamics
