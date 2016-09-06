#!/usr/bin/env python
""" Script to setup trimer for an interactive hoomd session"""
#
# Malcolm Ramsay 2016-03-09
#

# Hoomd helper functions

import math
import numpy as np
import hoomd
from hoomd import md
from hoomd import deprecated

# Let hoomd search for GPUs to use
hoomd.context.initialize()

# Create a 2D square lattice of 50x50 with particles of type A on the lattice
# points
system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4), n=[50, 50])
md.update.enforce2d()

# Set moments of inertia for every central particle
for particle in system.particles:
    if particle.type == 'A':
        particle.moment_inertia = (0, 0, 1.65)

# Let the system know there is anther type of particle
system.particles.types.add('B')

# Set interaction potentials for all interaction types
potentials = md.pair.lj(r_cut=2.5, nlist=md.nlist.cell())
potentials.pair_coeff.set('A', 'A', epsilon=1, sigma=2)
potentials.pair_coeff.set('B', 'B', epsilon=1, sigma=0.637556*2)
potentials.pair_coeff.set('A', 'B', epsilon=1, sigma=1.637556)

# Define rigid bodies
rigid = md.constrain.rigid()
# Each rigid body centered on the particles of type A
# has two particles of type B at the given relative coordinates
rigid.set_param('A', positions=[(math.sin(math.pi/3),
                                 math.cos(math.pi/3), 0),
                                (-math.sin(math.pi/3),
                                 math.cos(math.pi/3), 0)],
                types=['B', 'B']
               )
# Create the extra particles
rigid.create_bodies(create=True)
# A group only containing the center particles of the rigid bodies
center = hoomd.group.rigid_center()

# Output thermodynamic data
thermo = hoomd.analyze.log(
    filename="out.dat",
    quantities=[
        'temperature',
        'pressure',
        'volume',
        'translational_kinetic_energy',
        'rotational_kinetic_energy',
        'rotational_ndof',
        'translational_ndof',
        'N'
    ],
    period=1000
)

# Set integration parameters
md.integrate.mode_standard(dt=0.001)
npt = md.integrate.npt(
    group=center,
    kT=2.0,
    tau=5,
    P=13.5,
    tauP=5
)

hoomd.run(10000)

# Increase step size and decrease Nose-Hoover imaginary mass
md.integrate.mode_standard(dt=0.005)
npt.set_params(tau=1, tauP=1)

# Monitor simulation
xml = deprecated.dump.xml(filename="out.xml", group=hoomd.group.all(), all=True)
xml.write("out.xml")
hoomd.analyze.imd(
    port=4321,
    period=200,
)
hoomd.run(100000)
