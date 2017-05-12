"""Run a basic simulation"""

import hoomd
from hoomd import md
import molecule
import numpy as np
from StepSize import generate_steps
import TimeDep


def randomise(snapshot):
    """Randomise the momentum of the snapshot

    The randomisation is performed by shuffing the rotational and translational
    momenta independantly of each other. Where there are rigid bodies in the
    snapshot only the momenta of the centers are shuffled.

    Args:
        shapshot (class:`hoomd.data.snapshot`): Snapshot of which to shuffle
            momenta

    Returns:
        class:`hoomd.data.snapshot`: A new snapshot with randomised momenta
    """
    temp = hoomd.context.initialize('--mode=cpu --notice-level=0')
    with temp:
        system = hoomd.init.read_snapshot(snapshot)
        mysnap = system.take_snapshot(all=True)
    nbodies = mysnap.particles.body.max() + 1
    np.random.shuffle(mysnap.particles.velocity[:nbodies])
    np.random.shuffle(mysnap.particles.angmom[:nbodies])
    return mysnap


def initialise(snapshot):
    """Initialise a hoomd simulation"""
    context = hoomd.context.initialize('--mode=cpu --notice-level=0')
    with context:
        system = hoomd.init.read_snapshot(snapshot)
        md.update.enforce2d()
        mol = molecule.Trimer()
        mol.initialize(create=False)
        md.integrate.mode_standard(dt=0.005)
        md.integrate.npt(group=hoomd.group.rigid_center(),
                         kT=1.30, tau=1, P=13.50, tauP=1)
    return context, system


def read_snapshot(fname):
    """Read a hoomd snapshot from a hoomd gsd file

    Args:
        fname (string): Filename of GSD file to read in

    Returns:
        class:`hoomd.data.Snapshot`: Hoomd snapshot
    """
    with hoomd.context.initialize('--mode=cpu --notice-level=0'):
        system = hoomd.init.read_gsd(fname)
        return system.take_snapshot(all=True)


def main(init_file, steps):
    """Main function to run stuff"""
    snap1 = read_snapshot(init_file)
    snap2 = randomise(snap1)
    context1, sys1 = initialise(snap1)
    context2, sys2 = initialise(snap1)
    with context1:
        dynamics1 = TimeDep.TimeDep2dRigid(snap1, 0)
        for curr_step in generate_steps(steps):
            hoomd.run_upto(curr_step)
            dynamics1.append(sys1.take_snapshot(all=True), curr_step)
        fin1 = sys1.take_snapshot(all=True)
    with context2:
        dynamics2 = TimeDep.TimeDep2dRigid(snap2, 0)
        for curr_step in generate_steps(steps):
            hoomd.run_upto(curr_step)
            dynamics2.append(sys1.take_snapshot(all=True), curr_step)
        fin2 = sys1.take_snapshot(all=True)
    return (snap1, fin1, dynamics1.get_all_data(),
            snap2, fin2, dynamics2.get_all_data())


if __name__ == '__main__':
    main("./Trimer-13.50-1.30.gsd", 1000)
