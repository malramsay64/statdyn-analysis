"""Module for reading Translational and rotational data from a
    file to compute dynamics quantities
"""

import glob
from CompDynamics import CompRotDynamics
from TransData import TransRotData

def compute_file(fname, outfile='out.dat'):
    """Read data from file, compute dynamics and print results

    Args:
        fname (string): Path to file containing the translations and rotations
        outfile (string): Filename of output file
    """
    with open(fname) as infile:
        init = True
        for line in infile:
            data = TransRotData()
            data.from_json(line)
            dyn = CompRotDynamics(data)
            if init:
                dyn.print_heading(outfile)
                init = False
            dyn.print_all(outfile)

def compute_all(directory=".", pattern="*-tr.dat", suffix="-dyn.dat"):
    """Compute all files matching pattern"""
    files = glob.glob(directory+"/"+pattern)
    for infile in files:
        compute_file(infile, infile[:-(len(pattern)-1)]+suffix)
