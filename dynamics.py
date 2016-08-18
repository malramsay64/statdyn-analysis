#!/usr/bin/env python
""" A set of classes used for running simulations to compute the dynamic
properties of a Hoomd MD simulation"""

from __future__ import print_function
import os
import math
import hoomd
from hoomd import md
import StepSize
from TimeDep import TimeDep2dRigid
from CompDynamics import CompRotDynamics


def compute_dynamics(input_file,
                     temp,
                     press,
                     steps,
                    ):
    """Run a simulation computing the dynamic properties

    Run a hoomd simulation calculating the dynamic quantites on a power
    law scale such that both short timescale and long timescale events are
    vieable on the same figure while retaining a reasonable runtime.
    for the simulation.

    Args:
        input_file (string): Filename of the file containing the input
            configuration
        temp (float): The target temperature at which to run the simulation
        press (float): The target pressure at which to run the simulation
        steps (int): The number of steps for which to collect dynamics data
    """
    basename = os.path.splitext(input_file)[0]

    # Initialise simulation parameters
    hoomd.context.initialize()
    system = hoomd.init.read_gsd(filename=input_file, time_step=0)
    md.update.enforce2d()

    # Set moments of inertia for every central particle
    for particle in system.particles:
        if particle.type == 'A':
            particle.moment_inertia = (1.65, 10, 10)

    # Set interaction potentials
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

    # Set integration parameters
    md.integrate.mode_standard(dt=0.001)
    md.integrate.npt(group=center, kT=temp, tau=2, P=press, tauP=2)

    # initial run to settle system after reading file
    hoomd.run(100000)

    hoomd.analyze.log(filename=basename+"-thermo.dat",
                      quantities=['temperature', 'pressure',
                                  'potential_energy',
                                  'rotational_kinetic_energy',
                                  'translational_kinetic_energy'
                                 ],
                      period=1000)

    # Initialise dynamics quantities
    dyn = TimeDep2dRigid(system)
    CompRotDynamics().print_heading(basename+"-dyn.dat")
    tstep_init = system.get_metadata()['timestep']
    new_step = StepSize.PowerSteps(start=tstep_init)
    struct = [(new_step.next(), new_step, dyn)]
    timestep = tstep_init
    key_rate = 20000
    hoomd.dump.gsd(filename=basename+'.gsd',
                   period=10000000,
                   group=hoomd.group.all(),
                   overwrite=True,
                   truncate=True,
                  )

    while timestep < steps+tstep_init:
        index_min = struct.index(min(struct))
        next_step, step_iter, dyn = struct[index_min]
        timestep = next_step
        hoomd.run_upto(timestep)
        dyn.print_all(system, outfile=basename+"-dyn.dat")
        # dyn.print_data(system, outfile=basename+"-tr.dat")

        struct[index_min] = (step_iter.next(), step_iter, dyn)
        # Add new key frame every key_rate steps, limited to 5000
        if (timestep % key_rate == 0 and
                len(struct) < 5000 and
                len([s for s in struct if s[0] == timestep+1]) == 0):
            new_step = StepSize.PowerSteps(start=timestep)
            struct.append((new_step.next(), new_step, TimeDep2dRigid(system)))


