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
from statdyn import TimeDep, molecule
from statdyn.StepSize import generate_steps

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
    with hoomd.context.initialize(kwargs.get('init_args', '')):
        system = hoomd.init.read_snapshot(snapshot, time_step=0)
        md.update.enforce2d()
        mol = kwargs.get('mol', molecule.Trimer())
        mol.initialize(create=False)
        md.integrate.mode_standard(kwargs.get('dt', 0.005))
        md.integrate.npt(
            group=hoomd.group.rigid_center(),
            kT=temp,
            tau=kwargs.get('tau', 1.),
            P=kwargs.get('press', 13.5),
            tauP=kwargs.get('tauP', 1.)
        )
        dynamics = TimeDep.TimeDep2dRigid(system.take_snapshot(all=True), 0)
        for curr_step in generate_steps(steps):
            hoomd.run_upto(curr_step)
            dynamics.append(system.take_snapshot(all=True), curr_step)
        return dynamics.get_all_data()


def read_snapshot(fname, rand=False):
    """Read a hoomd snapshot from a hoomd gsd file

    Args:
        fname (string): Filename of GSD file to read in
        rand (bool): Whether to randomise the momenta of all the particles

    Returns:
        class:`hoomd.data.Snapshot`: Hoomd snapshot
    """
    if not hoomd.context.current:
        hoomd.context.initialize()
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
