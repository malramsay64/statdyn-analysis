#!/usr/bin/env python
""" A series of functions to equilibrate hoomd MD simulations"""
#
# Malcolm Ramsay 2016-03-09
#

# Hoomd helper functions

import os.path
from hoomd_script import (init, update, pair, group, integrate, analyze, run,
                          dump)

def equil_from_crys(input_xml="mol.xml",
                    outfile=None,
                    steps=100000,
                    potentials=None,
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
    # Ensure there is no configuration already initialised
    if init.is_initialized():
        init.reset()

    # Fix for an issue where the pressure equilibrates to a value larger than
    # the desired pressure
    press /= 2.2

    # Initialise simulation parameters
    # context.initialize()
    init.read_xml(filename=input_xml)
    update.enforce2d()

    if not potentials:
        potentials = pair.lj(r_cut=2.5)
        potentials.pair_coeff.set('1', '1', epsilon=1, sigma=2)
        potentials.pair_coeff.set('2', '2', epsilon=1, sigma=0.637556*2)
        potentials.pair_coeff.set('1', '2', epsilon=1, sigma=1.637556)

    # Create particle groups
    gall = group.all()

    # Find minimum energy configuration
    integrate.mode_minimize_rigid_fire(group=gall,
                                       dt=0.005,
                                       ftol=1e-3,
                                       Etol=1e-4
                                      )

    # Calculate thermodynamic quantities
    thermo = analyze.log(filename=None,
                         quantities=['temperature', 'pressure'],
                         period=1000
                        )

    # Perform initial equilibration at target temperature and pressure.
    integrate.mode_standard(dt=0.001)
    npt = integrate.npt_rigid(group=gall, T=temp, tau=5, P=press, tauP=5)


    iters = 0
    while (abs(thermo.query('temperature') - temp) > 0.1*temp or
           abs(thermo.query('pressure') - press) > 0.1*press):
        run(steps/10)
        iters += 1
        if iters > max_iters:
            break

    # Run longer equilibration
    integrate.mode_standard(dt=0.005)
    npt.set_params(tau=1, tauP=1)
    run(steps)

    # Write final configuration to file
    if not outfile:
        outfile = 'Trimer-{press}-{temp}'.format(press=press, temp=temp)
    xml = dump.xml(all=True)
    xml.write(filename=outfile)

def equil_from_file(input_xml=None,
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
    # Ensure there is no configuration already initialised
    if init.is_initialized():
        init.reset()

    # Fix for an issue where the pressure equilibrates to double the desired
    # pressure
    press /= 2.2

    # Initialise simulation parameters
    basename = os.path.splitext(outfile)[0]
    # context.initialize()
    init.read_xml(filename=input_xml, restart=basename+'-restart.xml')
    update.enforce2d()

    if not potentials:
        potentials = pair.lj(r_cut=2.5)
        potentials.pair_coeff.set('1', '1', epsilon=1, sigma=2)
        potentials.pair_coeff.set('2', '2', epsilon=1, sigma=0.637556*2)
        potentials.pair_coeff.set('1', '2', epsilon=1, sigma=1.637556)

    # Create particle groups
    gall = group.all()

    # Calculate thermodynamic quantities
    thermo = analyze.log(filename=basename+"-thermo.dat",
                         quantities=['temperature',
                                     'pressure',
                                     'potential_energy',
                                     'rotational_kinetic_energy',
                                     'translational_kinetic_energy'
                                    ],
                         period=1000
                        )

    # Create restart file if simulation stops
    dump.xml(filename=basename+'-restart.xml',
             period=1000000,
             restart=True,
             all=True)

    # Perform initial equilibration at target temperature and pressure.
    integrate.mode_standard(dt=0.001)
    npt = integrate.npt_rigid(group=gall, T=temp, tau=5, P=press, tauP=5)

    iters = 0
    while (abs(thermo.query('temperature') - temp) > 0.1*temp or
           abs(thermo.query('pressure') - press) > 0.1*press):
        run(10000)
        iters += 1
        if iters > max_iters:
            break

    # Run longer equilibration
    integrate.mode_standard(dt=0.005)
    npt.set_params(tau=1, tauP=1)
    run(steps)

    # Write final configuration to file
    if not outfile:
        outfile = 'Trimer-{press}-{temp}'.format(press=press, temp=temp)
    xml = dump.xml(all=True)
    xml.write(filename=outfile)



