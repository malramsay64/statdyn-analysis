#!/usr/bin/env python
""" Script to setup trimer for an interactive hoomd session"""
#
# Malcolm Ramsay 2016-03-09
#

# Hoomd helper functions

import os.path
import numpy as np
from hoomd_script import *
from TimeDep import TimeDep2dRigid

context.initialize()

if init.is_initialized():
    init.reset()

system = init.read_xml(filename="Trimer-13.50-5.00.xml")
update.enforce2d()

potentials = pair.lj(r_cut=2.5)
potentials.pair_coeff.set('1', '1', epsilon=1, sigma=2)
potentials.pair_coeff.set('2', '2', epsilon=1, sigma=0.637556*2)
potentials.pair_coeff.set('1', '2', epsilon=1, sigma=1.637556)


thermo = analyze.log(filename=None,\
                     quantities=['temperature', 'pressure'], \
                     period=1000 \
                    )
gall = group.all()


integrate.mode_standard(dt=0.001)
npt = integrate.npt_rigid(group=gall, \
                          T=2.5, \
                          tau=5, \
                          P=13.5/2, \
                          tauP=5\
                         )
run(100)
init = TimeDep2dRigid(system)
run(1000)
init.print_data(system)
run(1000)
init.print_data(system)
npt.set_params(tau=1, tauP=1)
run(10000)
integrate.mode_standard(dt=0.005)


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
        run(steps)
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
    npt.set_params(T=temp, P=press)
    run(steps)
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
