#!/usr/bin/env python
""" Equilibrate Trimer molecule at a series of temperatures
using Hoomd """

#
# Malcolm Ramsay 2016-03-09
#

# Runs a simulation of trimers

import os.path
import equil

PRESS = 13.5
# Holds tuples of the temperature and number of steps
# to iterate through
STEPS = 1000000
TEMPERATURES = [
    (5.00, 1*STEPS),
    (4.00, 1*STEPS),
    (3.50, 1*STEPS),
    (3.00, 1*STEPS),
    (2.50, 2*STEPS),
    (2.00, 2*STEPS),
    (1.80, 4*STEPS),
    (1.60, 4*STEPS),
    (1.50, 8*STEPS),
    (1.40, 8*STEPS),
    (1.35, 16*STEPS),
    (1.30, 16*STEPS),
]

if __name__ == "__main__":
    if not os.path.isfile("Trimer-init.gsd"):
        equil.equil_from_rand(outfile="Trimer-init.gsd", temp=0.1, press=PRESS)
    PREV_T = 5.00
    for temp, steps in TEMPERATURES:
        if temp == 5.00:
            input_file = "Trimer-init.gsd"
        else:
            input_file = ("Trimer-{press:.2f}-{temp:.2f}.gsd"
                          .format(press=PRESS, temp=PREV_T))
        PREV_T = temp
        outfile = ("Trimer-{press:.2f}-{temp:.2f}.gsd"
                   .format(press=PRESS, temp=temp))

        if not os.path.isfile(outfile):
            equil.equil_from_file(
                input_file=input_file,
                outfile=outfile,
                temp=temp,
                press=PRESS,
                steps=steps,
                max_iters=1
            )

