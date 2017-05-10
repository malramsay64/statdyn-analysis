"""Run a basic simulation"""

from copy import deepcopy as copy

import hoomd
import molecule
import numpy as np
from hoomd import md


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
        hoomd.init.read_snapshot(snapshot)
        md.update.enforce2d()
        mol = molecule.Trimer()
        mol.initialize(create=False)
        md.integrate.mode_standard(dt=0.005)
        md.integrate.npt(group=hoomd.group.rigid_center(),
                               kT=1.30, tau=1, P=13.50, tauP=1)
    return context

def read_snapshot(fname):
    with hoomd.context.initialize('--mode=cpu --notice-level=0'):
        system = hoomd.init.read_gsd(fname)
        return system.take_snapshot(all=True)

def main(init_file, steps):
    """Main function to run stuff"""
    snap1 = read_snapshot(init_file)
    snap2 = randomise(snap1)
    context1 = initialise(snap1)
    context2 = initialise(snap1)
    with context1:
        hoomd.run(steps)
        fin1 = hoomd.context.current.system_definition.takeSnapshot_float(*[True]*8)
    with context2:
        hoomd.run(steps)
        fin2 = hoomd.context.current.system_definition.takeSnapshot_float(*[True]*8)
    return (snap1, fin1, snap2, fin2)

if __name__ == '__main__':
    main("./Trimer-13.50-1.30.gsd", 1000)
