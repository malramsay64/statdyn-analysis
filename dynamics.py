#!/usr/bin/env python
""" A set of classes used for running simulations to compute the dynamic
properties of a Hoomd MD simulation"""

from __future__ import print_function
import os
import hoomd
from hoomd import md
import StepSize
from TimeDep import TimeDep2dRigid
from CompDynamics import CompRotDynamics
import molecule

MAX_FRAMES = 1000


def compute_dynamics(input_file, temp, press, steps, mol=None):
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

    if not mol:
        mol = molecule.Trimer()
    mol.initialise(create=False)
    center = hoomd.group.rigid_center()

    # Set integration parameters
    md.integrate.mode_standard(dt=0.001)
    md.integrate.npt(group=center, kT=temp, tau=1, P=press, tauP=1)

    # Zero momentum
    md.update.zero_momentum(period=10000)

    # initial run to settle system after reading file
    # hoomd.run(100000)
    md.integrate.mode_standard(dt=0.005)

    hoomd.analyze.log(filename=basename+"-thermo.dat",
                      quantities=[
                          'temperature',
                          'pressure',
                          'potential_energy',
                          'volume',
                          'N',
                          'rotational_kinetic_energy',
                          'translational_kinetic_energy'
                      ],
                      period=1000)

    # Initialise dynamics quantities
    snapshot = system.take_snapshot()
    tstep_init = hoomd.get_step()
    timestep = tstep_init
    last_timestep = tstep_init
    dyn = TimeDep2dRigid(snapshot, timestep)
    CompRotDynamics().print_heading(basename+"-dyn.dat")
    new_step = StepSize.PowerSteps(num_linear=19, start=tstep_init)
    struct = [(new_step.next(), new_step, dyn)]
    key_rate = 20000
    gsd = hoomd.dump.gsd(
        filename=basename+'-traj.gsd',
        period=None,
        group=hoomd.group.all(),
        overwrite=True,
        truncate=False,
        static=['attribute', 'topology']
    )

    while timestep < steps+tstep_init:
        index_min = struct.index(min(struct))
        next_step, step_iter, dyn = struct[index_min]
        timestep = min(next_step, steps)
        # Only run if incrementing number of steps
        if timestep > last_timestep:
            hoomd.run_upto(timestep)
            gsd.write_restart()
            snapshot = system.take_snapshot()

        last_timestep = timestep
        dyn.print_all(snapshot, timestep, outfile=basename+"-dyn.dat")
        # dyn.print_data(system, outfile=basename+"-tr.dat")
        struct[index_min] = (step_iter.next(), step_iter, dyn)
        # Add new key frame every key_rate steps, limited to 5000
        if (timestep % key_rate == 0 and
                len(struct) < MAX_FRAMES and
                len([s for s in struct if s[0] == timestep+1]) == 0):
            new_step = StepSize.PowerSteps(num_linear=19, start=timestep)
            struct.append(
                (new_step.next(), new_step, TimeDep2dRigid(snapshot, timestep))
            )
