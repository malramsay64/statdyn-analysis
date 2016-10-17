"""Module for reading Translational and rotational data from a
    file to compute dynamics quantities.
"""

import glob
import gsd.fl
import gsd.hoomd
from TimeDep import TimeDep2dRigid
from CompDynamics import CompRotDynamics
# from TransData import TransRotData


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
    infile = gsd.fl.GSDFile(fname, 'rb')
    snapshots = gsd.hoomd.HOOMDTrajectory(infile)
    keyframes = []
    for i, snapshot in enumerate(snapshots):
        if i == 0:
            CompRotDynamics().print_heading(outfile)
        for frame in keyframes:
            frame.print_all(snapshot, snapshot.configuration.step, outfile)
        if i % 10 == 0:
            keyframes.append(TimeDep2dRigid(snapshot, snapshot.configuration.step))


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
        print(infile)
        compute_file(infile, infile[:-(len(pattern)-1)]+suffix)

if __name__ == "__main__":
    # compute_file("Trimer-13.50-5.00-traj.gsd")
    compute_all(pattern="*0-tr.dat", suffix="0-dyn.dat")
