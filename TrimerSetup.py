#!/usr/bin/env python
""" Script to setup trimer for an interactive hoomd session"""
#
# Malcolm Ramsay 2016-03-09
#

# Hoomd helper functions

import os.path
import math
import numpy as np
import hoomd
from hoomd import md
from TimeDep import TimeDep2dRigid

hoomd.context.initialize()

input_file = 'Trimer-13.50-5.00.gsd'
system = hoomd.init.read_gsd(filename=input_file, time_step=0)
md.update.enforce2d()
# Set moments of inertia for every central particle
for particle in system.particles:
    if particle.type == 'A':
        particle.moment_inertia = (1.65, 10, 10)


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


thermo = hoomd.analyze.log(filename=None,\
                     quantities=['temperature', 'pressure'], \
                     period=1000 \
                    )

md.integrate.mode_standard(dt=0.001)
npt = md.integrate.npt(group=center, \
                          kT=2.5, \
                          tau=5, \
                          P=13.5/2, \
                          tauP=5\
                         )
hoomd.run(100)
init = TimeDep2dRigid(system)
npt.set_params(tau=1, tauP=1)
hoomd.run(10000)
md.integrate.mode_standard(dt=0.005)


def get_stats(steps=1000, points=200):
    """ Calculate statistics for the pressure and temperature
    of the simulation and compare them to the expected values.
    param: steps The number of steps between points, defults
    to 1000
    param: point The number of points to collect for statistics,
    defaults to 200
    """
    press = []
    temp = []
    for i in range(points):
        hoomd.run(steps)
        temp.append(thermo.query('temperature'))
        press.append(thermo.query('pressure'))
    print("Temp -- Mean: {mean} Stddev: {stdev}"\
            .format(mean=np.mean(temp), stdev=np.std(temp)))
    print("Press -- Mean: {mean} Stddev: {stdev}"\
            .format(mean=np.mean(press), stdev=np.std(press)))
    return (np.mean(temp), \
            np.std(temp),  \
            np.mean(press),\
            np.std(press)  \
           )

snapshot = ''
def run_t_p(temp, press, steps=500000):
    npt.set_params(kT=temp, P=press)
    hoomd.run(steps)
    stats = get_stats()
    print(stats)
    global snapshot
    snapshot = system.take_snapshot(all=True)
    return stats


vals = [(5.0, 13.5),
        (4.0, 10.0),
        (3.0, 8.5),
        (4.0, 6.75),
        (3.0, 6.75),
        (2.5, 6.75),
        (2.0, 6.75),
        (1.8, 6.75),
        (2.5, 5.0),
        (2.0, 4.0),
        (1.5, 3.0),
       ]
