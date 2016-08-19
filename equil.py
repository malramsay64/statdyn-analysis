#!/usr/bin/env python
"""This module contains a series of functions to easily equilibrate hoomd
MD simulations. """
#
# Malcolm Ramsay 2016-03-09
#

# Hoomd helper functions

import os.path
import math
import hoomd
from hoomd import md

def equil_from_rand(outfile=None,
                    steps=100000,
                    temp=1.0,
                    press=1.0,
                    max_iters=10):
    R""" Equilibrate system from a lattice initial configuration

    The inttial configuration consists of a numer of particles on a hexagonal
    lattice arrangement. Each particle is the central atom of a molecule which
    is currently a bent trimer molecule.

    The equilibration is initally carried out at low temperature with small
    timesteps allowing any high energy overlaps to be dealt with gracefully.

    Todo:
        - Pass molecule parameters to the function allowing simple expansion to
          any molecular shape.
        - Pass the pair potentials which is relevant to the previous point in
          being able to easily expand to different molecules.
        - Pass the number of molecules as a paramter
        - Compute moments of interta of molecules or allow them to be passed to
          the function

    Args:
        outfile (string): Filename of file to output final configuration to.
            The output file will be in the GSD file format and so it is
            advisable the output files have the `.gsd` extension.
        steps (int): Number of steps to run the equilibration.
        temp (float): Target temperature at which to equilibrate system
        press (float): Target pressure for equilibration
        max_iters (int): Maximum number of iterations for convergence on
            target pressure and temperature.

    """
    # Initialise context, also removes any previously intialised contexts
    hoomd.context.initialize()

    # Create hexagonal lattice of central particles
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.hex(a=4),
                                       n=[25, 25])

    # Assign the moment of intertial of each molecule.
    for particle in system.particles:
        particle.moment_inertia = (1.65, 10, 10)

    system.particles.types.add('B')

    # Create the pair coefficients
    lj_c = md.pair.lj(r_cut=2.5, nlist=md.nlist.cell())
    lj_c.pair_coeff.set('A', 'A', epsilon=1.0, sigma=2.0)
    lj_c.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.637556)
    lj_c.pair_coeff.set('B', 'B', epsilon=1.0, sigma=2*0.637556)

    # Create rigid particles and define their configuration
    rigid = md.constrain.rigid()
    rigid.set_param('A', positions=[(math.sin(math.pi/3),
                                     math.cos(math.pi/3), 0),
                                    (-math.sin(math.pi/3),
                                     math.cos(math.pi/3), 0)],
                    types=['B', 'B']
                   )
    rigid.create_bodies(create=True)
    center = hoomd.group.rigid_center()
    md.update.enforce2d()

    # Calculate thermodynamic quantities
    thermo = hoomd.analyze.log(filename=None,
                               quantities=['temperature', 'pressure'],
                               period=1000
                              )

    # Perform initial equilibration at target temperature and pressure.
    md.integrate.mode_standard(dt=0.001)
    npt = md.integrate.npt(group=center, kT=temp, tau=5, P=press, tauP=5)

    # Iterate  until the target pressure and temperature are close to the
    # target values.
    iters = 0
    while (abs(thermo.query('temperature') - temp) > 0.1*temp or
           abs(thermo.query('pressure') - press) > 0.1*press):
        hoomd.run(steps/10)
        iters += 1
        if iters > max_iters:
            break

    # Run longer equilibration
    md.integrate.mode_standard(dt=0.005)
    npt.set_params(tau=1, tauP=1)
    hoomd.run(steps)

    # Write final configuration to file
    if not outfile:
        outfile = 'Trimer-{press}-{temp}.gsd'.format(press=press, temp=temp)
    hoomd.dump.gsd(filename=outfile,
                   period=None,
                   group=hoomd.group.all())

def equil_from_file(input_file=None,
                    outfile=None,
                    temp=1.0,
                    press=1.0,
                    steps=100000,
                    max_iters=10,
                    potentials=None):
    """ Equilibrate simulation from an input file

    This function is to equilibrate configuration for a number of timesteps
    such that the configuration has reached thermodynamic equilibrium.

    Args:
        input_file (string): Input file containing input configuration in
            the  GSD format
        outfile (string): File to write equilibrated configuration to. This
            output file will be in the GSD file format and the filename should
            include the `.gsd` extension.
        temp (float): Target temperature for equilibration
        press (float): Target pressure for equilibration
        steps (int): Number of steps to run equilibration
        max_iters (int): Maximum number of iterations to run equilibration step
            if target temperature or pressure are not reached.
        potentials (:class:`md.pair`): Custom interaction potentials for the
            simulation
    """
    # Initialise simulation parameters
    basename = os.path.splitext(outfile)[0]
    hoomd.context.initialize()
    hoomd.init.read_gsd(filename=input_file, time_step=0)
    md.update.enforce2d()

    # Set interaction potentials
    if not potentials:
        potentials = md.pair.lj(r_cut=2.5, nlist=md.nlist.cell())
        potentials.pair_coeff.set('A', 'A', epsilon=1, sigma=2)
        potentials.pair_coeff.set('B', 'B', epsilon=1, sigma=0.637556*2)
        potentials.pair_coeff.set('A', 'B', epsilon=1, sigma=1.637556)

    # Set configuration of rigid bodies
    rigid = md.constrain.rigid()
    rigid.set_param('A', positions=[(math.sin(math.pi/3),
                                     math.cos(math.pi/3), 0),
                                    (-math.sin(math.pi/3),
                                     math.cos(math.pi/3), 0)],
                    types=['B', 'B']
                   )
    rigid.create_bodies(create=False)
    center = hoomd.group.rigid_center()

    # Calculate thermodynamic quantities
    thermo = hoomd.analyze.log(filename=basename+"-thermo.dat",
                               quantities=['temperature',
                                           'pressure',
                                           'potential_energy',
                                           'rotational_kinetic_energy',
                                           'translational_kinetic_energy'
                                          ],
                               period=1000
                              )

    # Perform initial equilibration at target temperature and pressure.
    md.integrate.mode_standard(dt=0.001)
    npt = md.integrate.npt(group=center, kT=temp, tau=5, P=press, tauP=5)

    # Equilibrate until close to target thermodynamic parameters
    iters = 0
    while (abs(thermo.query('temperature') - temp) > 0.1*temp or
           abs(thermo.query('pressure') - press) > 0.1*press):
        hoomd.run(10000)
        iters += 1
        if iters > max_iters:
            break

    # Run longer equilibration
    md.integrate.mode_standard(dt=0.005)
    npt.set_params(tau=1, tauP=1)
    hoomd.run(steps)

    # Write final configuration to file
    if not outfile:
        outfile = 'Trimer-{press}-{temp}.gsd'.format(press=press, temp=temp)
    hoomd.dump.gsd(filename=outfile,
                   period=None,
                   group=hoomd.group.all())



