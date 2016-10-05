#!/bin/env python3
#
# copyright Malcolm Ramsay
#
"""Module to define a molecule to use for simulation."""

import math
import hoomd
import hoomd.md


class Molecule(object):
    """Molecule class holding all the information on the molecule for use
    in hoomd.

    This class contains all the paramters required to initialise the molecule
    in a hoomd simulation.
    """
    def __init__(self):
        self._system = hoomd.context.current.system_definition
        self.moment_inertia = (0, 0, 0)
        self.potential = hoomd.md.pair.lj
        self.potential_args = dict(r_cut=2.5, nlist=hoomd.md.nlist.cell())
        self.particles = []

    def initialise(self, create=False):
        """Initialse the molecule for hoomd to use"""
        pass

    def set_potential(self, potential, args):
        """Set the interaction potential of the molecules.

        Args:
            potential (class:`hoomd.md.pair`): Interaction potential of the
                molecules
            args (dict): dict containing all the rguments for the
                interaction potential.
        """
        self.potential = potential
        self.potential_args = args

    def define_particles(self):
        """Add the particles to the simulation context"""
        for particle in self.particles:
            self._system.getParticleData().addType(particle)

    def define_potential(self):
        """Define the potential in the simulation context"""
        self.potential(**self.potential_args)

    def define_rigid(self, create=False):
        """Define the rigid constraints of the molecule

        Args:
            create (bool): Flag that toggles the option of creating the
                additional particles when creating the rigid bodies. Defaults
                to False.
        """
        pass

    def define_moment_inertia(self):
        """Set the moment of intertia of all particles in the system"""
        mom_i = hoomd._hoomd.Scalar3()
        mom_i.x, mom_i.y, mom_i.z = self.moment_inertia
        for particle in range(self._system.getParticleData().getN()):
            self._system.getParticleData().setMomentsOfInertia(particle, mom_i)


class Trimer(Molecule):
    """Initialises a trimer molecule

    Args:
        radius (float): Radius of the small particles. Default is 0.637556
        distance (float): Distance of the outer particles from the central
            one. Default is 1.0
        angle (float): Angle between the two outer particles in degrees.
            Default is 120
    """
    def __init__(self, radius=0.637556, distance=1.0, angle=120):
        super(Trimer, self).__init__()
        self.radius = radius
        self.distance = distance
        self.angle = angle
        self.particles = ["B"]
        self.moment_inertia = (0, 0, 1.65)

    def define_potential(self):
        """Define the potential in the simulation context"""
        potential = self.potential(**self.potential_args)
        potential.pair_coeff.set('A', 'A', epsilon=1, sigma=2.0)
        potential.pair_coeff.set('B', 'B', epsilon=1, sigma=self.radius*2)
        potential.pair_coeff.set('A', 'B', epsilon=1, sigma=1.0+self.radius)

    def define_rigid(self, create=False):
        """Define the rigid constraints of the molecule

        Args:
            create (bool): Flag that toggles the option of creating the
                additional particles when creating the rigid bodies. Defaults
                to False.
        """
        angle = (self.angle/2)*math.pi/180.
        rigid = hoomd.md.constrain.rigid()
        rigid.set_param(
            'A',
            positions=[
                (math.sin(angle), math.cos(angle), 0),
                (-math.sin(angle), math.cos(angle), 0)
            ],
            types=['B', 'B']
        )

    def initialise(self, create=False):
        """Initialise the molecule such that hoomd can use it for
        simulations"""
        self.define_particles()
        self.define_potential()
        self.define_moment_inertia()
        self.define_rigid(create)


if __name__ == "__main__":
    hoomd.context.initialize()
    hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4), n=[4, 4])
    Trimer().initialise()
