#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Module for setting up and running a hoomd simulation
"""
import os
import hoomd
from hoomd import md
import numpy as np
import pandas
from statdyn import TimeDep, initialise
from statdyn.StepSize import generate_steps, generate_step_series


def set_defaults(kwargs):
    kwargs.setdefault('tau', 1.)
    kwargs.setdefault('press', 13.5)
    kwargs.setdefault('tauP', 1.)
    kwargs.setdefault('dt', 0.005)
    kwargs.setdefault('thermo', True)
    kwargs.setdefault('thermo_dir', '.')
    kwargs.setdefault('thermo_period', 10000)


def run_npt(snapshot, temp, steps, **kwargs):
    """Initialise and run a hoomd npt simulation

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
    context = hoomd.context.initialize('')
    set_defaults(kwargs)
    kwargs['context'] = context
    kwargs['temp'] = temp
    print(kwargs.get('tauP'))
    with context:
        system = initialise.init_from_snapshot(snapshot, **kwargs)
        _set_integrator(kwargs)
        _set_thermo(kwargs)
        dynamics = TimeDep.TimeDep2dRigid(system.take_snapshot(all=True), 0)
        for curr_step in generate_steps(steps):
            hoomd.run_upto(curr_step)
            dynamics.append(system.take_snapshot(all=True), curr_step)
    return dynamics.get_all_data()


def run_multiple_concurrent(snapshot, temp, steps, **kwargs):
    """Initialise and run a hoomd npt simulatins with data collection for
    dynamics.

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
    context = hoomd.context.SimulationContext()
    set_defaults(kwargs)
    kwargs['context'] = context
    kwargs['temp'] = temp
    with context:
        system = initialise.init_from_snapshot(snapshot, **kwargs)
        _set_integrator(kwargs)
        _set_thermo(kwargs)
        dynamics = TimeDep.TimeDepMany()
        dynamics.add_init(system.take_snapshot(all=True), 0, 0)
        for curr_step, index in generate_step_series(steps, index=True):
            hoomd.run_upto(curr_step)
            dynamics.append(system.take_snapshot(all=True), index, curr_step)
    return dynamics.get_data()


def _set_integrator(kwargs):
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
            kwargs.get('thermo_dir')+'/'+'thermo-{press}-{temp}.log'.format(
                press=kwargs.get('press'), temp=kwargs.get('temp')),
            ['volume', 'potential_energy', 'kinetic_energy',
             'rotational_kinetic_energy', 'temperature', 'pressure'],
            period=kwargs.get('thermo_period'),
        )


def read_snapshot(fname, rand=False):
    """Read a hoomd snapshot from a hoomd gsd file

    Args:
        fname (string): Filename of GSD file to read in
        rand (bool): Whether to randomise the momenta of all the particles

    Returns:
        class:`hoomd.data.Snapshot`: Hoomd snapshot
    """
    with hoomd.context.initialize(''):
        snapshot = hoomd.data.gsd_snapshot(fname)
        if rand:
            nbodies = snapshot.particles.body.max() + 1
            np.random.shuffle(snapshot.particles.velocity[:nbodies])
            np.random.shuffle(snapshot.particles.angmom[:nbodies])
            return snapshot


def iterate_random(directory, temp, steps, iterations=2, **kwargs):
    """Main function to run stuff
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
    init_file = directory + "/Trimer-{press:.2f}-{temp:.2f}.gsd".format(
        press=kwargs.get('press', 13.5),
        temp=temp
    )
    for iteration in range(iterations):
        dynamics = run_npt(
            read_snapshot(init_file, rand=True),
            temp,
            steps,
            **kwargs
        )
        with pandas.HDFStore(os.path.splitext(init_file)[0]+'.hdf5') as store:
            store['dyn{i}'.format(i=iteration)] = dynamics
