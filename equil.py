#!/usr/bin/env python
""" A series of functions to equilibrate hoomd MD simulations"""
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
    """ Equilibrate system from a crystalline initial configuration

    Args:
        input_xml (string): Filename of file in which the initial configuration
            is stored in the hoomd-xml file format.
        outfile (string): Filename of file to output final configuration to.
        steps (int): Number of steps to run the equilibration.
        potentials (class:`hoomd.pair`): Interaction potentials to use for the
            simulation. Default values are set if no input is given.
        temp (float): Target temperature at which to equilibrate system
        press (float): Target pressure for equilibration
        max_iters (int): Maximum number of iterations for convergence on
            target pressure and temperature.

    """
    hoomd.context.initialize()

    # Create hexagonal lattice of central particles
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.hex(a=4),
                                       n=[25, 25])

    for particle in system.particles:
        particle.moment_inertia = (1.65, 10, 10)

    system.particles.types.add('B')

    lj_c = md.pair.lj(r_cut=2.5, nlist=md.nlist.cell())
    lj_c.pair_coeff.set('A', 'A', epsilon=1.0, sigma=2.0)
    lj_c.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.637556)
    lj_c.pair_coeff.set('B', 'B', epsilon=1.0, sigma=2*0.637556)

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

    Args:
        input_xml (string): Input file containing input configuration in
            hoomd-xml format
        outfile (string): File to write equilibrated configuration to
        temp (float): Target temperature for equilibration
        press (float): Target pressure for equilibration
        steps (int): Number of steps to run equilibration
        max_iters (int): Maximum number of iterations to run equilibration step
            if target temperature or pressure are not reached.
        potentials (:class:`hoomd.pair`): Custom interaction potentials for the
            simulation
    """
    # Initialise simulation parameters
    basename = os.path.splitext(outfile)[0]
    hoomd.context.initialize()
    hoomd.init.read_gsd(filename=input_file, time_step=0)
    md.update.enforce2d()

    if not potentials:
        potentials = md.pair.lj(r_cut=2.5, nlist=md.nlist.cell())
        potentials.pair_coeff.set('A', 'A', epsilon=1, sigma=2)
        potentials.pair_coeff.set('B', 'B', epsilon=1, sigma=0.637556*2)
        potentials.pair_coeff.set('A', 'B', epsilon=1, sigma=1.637556)


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



