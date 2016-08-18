"""Module for reading Translational and rotational data from a
    file to compute dynamics quantities.
"""

import glob
from CompDynamics import CompRotDynamics
from TransData import TransRotData

def compute_file(fname, outfile='out.dat'):
    """Read data from file, compute dynamics and print results

    The file is read line by line, where the data from each frame should
    reside on a single line. While not sticking completely to the JSON format
    it allows for file sizes larger than would fit into memory.

    Note:
        This assumes that the data being read from a file is a
        class:`TransData.TransRotData` object.

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

def compute_all(pattern="*-tr.dat", suffix="-dyn.dat", directory="."):
    """Compute all files in directory matching pattern

    This finds all the files in a specified directory that match a pattern,
    computing all the dynamic quanities on the translational and rotational
    data in that file.

    Todo:
        * Spawn the analysis into multiple threads

    Args:
        pattern (str): Pattern that matches the files to compute. Defaults to
            `"*-tr.dat"`.
        suffix (str): replace the last characters of the file with this.
            Default is `"-dyn.dat"`.
        directory (str): The directory in which to search for the files.
            Default is the current directory.
    """
    files = glob.glob(directory+"/"+pattern)
    for infile in files:
        compute_file(infile, infile[:-(len(pattern)-1)]+suffix)

if __name__ == "__main__":
    compute_all()
