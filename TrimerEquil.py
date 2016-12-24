#!/usr/bin/env python
""" Equilibrate Trimer molecule at a series of temperatures
using Hoomd """

#
# Malcolm Ramsay 2016-03-09
#

# Runs a simulation of trimers

import os.path
from sys import argv
import equil
import crystals

PRESS = 13.5
# Holds tuples of the temperature and number of steps
# to iterate through
STEPS = 10000000
TEMPERATURES = {
    5.00: 1*STEPS,
    4.00: 1*STEPS,
    3.50: 1*STEPS,
    3.00: 1*STEPS,
    2.50: 2*STEPS,
    2.00: 4*STEPS,
    1.80: 8*STEPS,
    1.60: 8*STEPS,
    1.50: 16*STEPS,
    1.40: 32*STEPS,
    1.35: 64*STEPS,
    1.30: 128*STEPS,
    1.25: 256*STEPS,
    1.20: 512*STEPS,
    1.15: 1024*STEPS,
    1.10: 2048*STEPS,
}


def main(unitcell=None):
    """Main function to run equilibration of hoomd files"""
    # Equilibrate initial file
    if not os.path.isfile("Trimer-init.gsd"):
        equil.equil_from_lattice(
            outfile="Trimer-init.gsd",
            temp=0.1,
            press=PRESS,
            unitcell=unitcell
        )
    prev_t = 5.00
    # If argument passed
    if len(argv) == 2:
        if TEMPERATURES.get(argv[1], 0):
            temp = float(argv[2])
            for i in sorted(TEMPERATURES.keys(), reverse=True):
                if temp == i:
                    break
                prev_t = i
            steps = TEMPERATURES.get(temp)
            if temp == 5.00:
                input_file = "Trimer-init.gsd"
            else:
                input_file = ("Trimer-{press:.2f}-{temp:.2f}.gsd"
                              .format(press=PRESS, temp=prev_t))
            equil.equil_from_file(
                input_file=input_file,
                outfile=outfile,
                temp=temp,
                press=PRESS,
                steps=steps,
                max_iters=1
            )
    # Default behaviour, equilibrate everything
    for temp, steps in sorted(TEMPERATURES.items(), reverse=True):
        if temp == 5.00:
            input_file = "Trimer-init.gsd"
        else:
            input_file = ("Trimer-{press:.2f}-{temp:.2f}.gsd"
                          .format(press=PRESS, temp=prev_t))
        prev_t = temp
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


if __name__ == "__main__":
    unitcell=crystals.p2().get_unitcell()
    main(unitcell)
