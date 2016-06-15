#!/usr/bin/env python3
""" A series of functions to equilibrate hoomd MD simulations"""
#
# Malcolm Ramsay 2016-03-09
#

# Hoomd helper functions

import os.path
from hoomd_script import analyze,  \
                         run,      \
                         init,     \
                         pair,     \
                         update,   \
                         group,    \
                         integrate,\
                         dump

def equil_from_crys(input_xml="mol.xml",\
                  outfile=None,\
                  steps=100000,\
                  rigid=True,\
                  potentials=None,\
                  temp=1.0,\
                  press=1.0,\
                  max_iters=10):
    """ Equilibrate system from a crystalline initial configuration
    :param inputXML: File in which the initial configuration is stored in the
    hoomd-xml file format.
    :param outfile: File to output final configuration to.
    :param steps: Number of steps to run the equilibration.
    :param rigid: Boolean value indicating whether the rigid integrators should
    be used.
    :param potentials: Interaction potentials to use for the simulation. Default
    values are set if no input is given.
    :param temp: Target temperature at which to equilibrate system
    :param press: Target pressure for equilibration
    :param maxIters: Maximum number of iterations for convergence on target
    pressure and temperature.
    """
    # Ensure there is no configuration already initialised
    if init.is_initialized():
        init.reset()

    # Fix for an issue where the pressure equilibrates to a value larger than
    # the desired pressure
    press /= 2.2

    # Initialise simulation parameters
    # context.initialize()
    system = init.read_xml(filename=input_xml)
    update.enforce2d()

    if not potentials:
        potentials = pair.lj(r_cut=2.5)
        potentials.pair_coeff.set('1', '1', epsilon=1, sigma=2)
        potentials.pair_coeff.set('2', '2', epsilon=1, sigma=0.637556*2)
        potentials.pair_coeff.set('1', '2', epsilon=1, sigma=1.637556)

    # Create particle groups
    gall = group.all()

    # Find minimum energy configuration
    if rigid:
        fire = integrate.mode_minimize_rigid_fire(group=gall,\
                                                  dt=0.005,  \
                                                  ftol=1e-3, \
                                                  Etol=1e-4  \
                                                 )
    else:
        fire = integrate.mode_minimize_fire(group=gall,\
                                            dt=0.005,  \
                                            ftol=1e-3, \
                                            Etol=1e-4  \
                                           )
    run(1000)
    while not fire.has_converged():
        run(1000)

    # Calculate thermodynamic quantities
    thermo = analyze.log(filename=None, \
                         quantities=['temperature', 'pressure'], \
                         period=1000 \
                        )

    # Perform initial equilibration at target temperature and pressure.
    integrate.mode_standard(dt=0.001)
    if rigid:
        npt = integrate.npt_rigid(group=gall, T=temp, tau=5, P=press, tauP=5)
    else:
        npt = integrate.npt(group=gall, T=temp, tau=5, P=press, tauP=5)


    iters = 0
    while abs(thermo.query('temperature') - temp) > 0.1*temp or \
          abs(thermo.query('pressure') - press) > 0.1*press:
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

def equil_from_file(input_xml=None,\
                    outfile=None,\
                    temp=1.0,\
                    press=1.0,\
                    steps=100000,\
                    max_iters=10,\
                    rigid=True,\
                    potentials=None):
    """ Equilibrate simulation from an input file

    :param inputXml: Input file containing input configuration in hoomd-xml
    format
    :param outfile: File to write equilibrated configuration to
    :param temp: Target temperature for equilibration
    :param press: Target pressure for equilibration
    :param steps: Number of steps to run equilibration
    :param maxIters: Maximum number of iterations to run equilibration step if
    target temperature or pressure are not reached.
    :param potentials: Custom interaction potentials for the simulation
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
    system = init.read_xml(filename=input_xml, restart=basename+'-restart.xml')
    update.enforce2d()

    if not potentials:
        potentials = pair.lj(r_cut=2.5)
        potentials.pair_coeff.set('1', '1', epsilon=1, sigma=2)
        potentials.pair_coeff.set('2', '2', epsilon=1, sigma=0.637556*2)
        potentials.pair_coeff.set('1', '2', epsilon=1, sigma=1.637556)

    # Create particle groups
    gall = group.all()

    # Calculate thermodynamic quantities
    thermo = analyze.log(filename=basename+"-thermo.dat", \
                         quantities=['temperature', \
                                     'pressure', \
                                     'potential_energy', \
                                     'rotational_kinetic_energy', \
                                     'translational_kinetic_energy'\
                                    ], \
                        period=1000\
                        )

    # Create restart file if simulation stops
    restart = dump.xml(filename=basename+'-restart.xml', \
                       period=1000000, \
                       restart=True, \
                       all=True)

    # Perform initial equilibration at target temperature and pressure.
    integrate.mode_standard(dt=0.001)
    if rigid:
        npt = integrate.npt_rigid(group=gall, T=temp, tau=5, P=press, tauP=5)
    else:
        npt = integrate.npt(group=gall, T=temp, tau=5, P=press, tauP=5)

    iters = 0
    while abs(thermo.query('temperature') - temp) > 0.1*temp or \
          abs(thermo.query('pressure') - press) > 0.1*press:
        run(10000)
        iters += 1
        if iters > maxIters:
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



