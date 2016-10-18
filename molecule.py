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
    in a hoomd simulation. This includes all interaction potentials, the rigid
    body interactions and the moments of inertia.

    The Molecule class is a template class that defines a number of functions
    subclasses can use to set these variables however it generates no sensible
    molecule itself.
    """
    def __init__(self):
        self.moment_inertia = (0, 0, 0)
        self.potential = hoomd.md.pair.lj
        self.potential_args = dict()
        self.particles = []

    def initialise(self, create=False):
        """Initialse the molecule for hoomd to use

        func:`Molecule.initialise` initialises all the molecule variables
        within the current hoomd context.

        Args:
            create (bool): Boolean flag to indicate whether to create the
                particles surrounding the center of the rigid bodies.
                Default is False.
        """
        self._system = hoomd.context.current.system_definition
        self.define_particles()
        self.define_moment_inertia()
        self.define_potential()
        self.define_rigid(create)

    def initialize(self, create):
        """Because spelling. See func:`Molecule.initialise`"""
        return self.initialise(create)

    def set_potential(self, potential, args):
        """Set the interaction potential of the molecules.

        Args:
            potential (class:`hoomd.md.pair`): Interaction potential of the
                molecules
            args (dict): dict containing the arguments for the
                interaction potential.

        Note:
            The `nlist` property doesn't work properly when initialised
            from a dictionary. Not entirely sure why, but avoid including that
            value when passing the arguments
        """
        self.potential = potential
        self.potential_args = args

    def define_particles(self):
        """Add the particles to the simulation context

        A helper function that adds the extra particles required for the
        molecule to the hoomd simulation context.
        """
        for particle in self.particles:
            self._system.getParticleData().addType(particle)

    def define_potential(self):
        R"""Define the potential in the simulation context

        A helper function that defines the potential to be used by the  hoomd
        simulation context. The default values for the potential are a
        Lennard-Jones potential with a cutoff of 2.5 and interaction parameters
        of :math:`\epsilon = 1.0` and :math:`\sigma = 2.0`.

        Returns:
            class:`hoomd.md.pair`: The interaction potential object class.
        """
        self.potential_args.setdefault('r_cut', 2.5)
        potential = self.potential(
            **self.potential_args,
            nlist=hoomd.md.nlist.cell()
        )
        potential.pair_coeff.set('A', 'A', epsilon=1, sigma=2.0)
        return potential

    def define_rigid(self, create=False, params=None):
        """Define the rigid constraints of the molecule

        This is a helper function to define the rigid body constraints of the
        particular molecules within the hoomd context.

        Args:
            create (bool): Flag that toggles the option of creating the
                additional particles when creating the rigid bodies. Defaults
                to False.
            params (dict): Dictionary defining the rigid body structure. The
                default values for the `type_name` of A and the `types` of the
                `self.particles` variable should work for the vast majority of
                systems, so the only value required should be the topology.

        Returns:
            class:`hoomd.md.constrain.rigid`: Rigid constraint object
        """
        if not params:
            params = dict()
        params.setdefault('type_name', 'A')
        params.setdefault('types', self.particles)
        rigid = hoomd.md.constrain.rigid()
        rigid.set_param(**params)
        rigid.create_bodies(create)
        return rigid

    def define_moment_inertia(self):
        """Set the moment of intertia of all particles in the system

        A helper function to set the moment of inertia of all the molecules in
        the current hoomd context. It sets the moment of inertia of all
        all particles in the simulation system to the value of the variable
        `self.moment_inertia`.

        Note:
            This changes the moment of intertia of every particle in the
            system. As initialise is currently set up the moments of intertia
            are set before the extra particles in the rigid bodies are
            created. While I don't expect there to be issues with running this
            after those particles have been created I currently don't know
            the outcome, so beware.
        """
        mom_i = hoomd._hoomd.Scalar3()
        mom_i.x, mom_i.y, mom_i.z = self.moment_inertia
        for particle in range(self._system.getParticleData().getN()):
            self._system.getParticleData().setMomentsOfInertia(particle, mom_i)

    def set_moment_inertia(self, moment_inertia):
        """Set the moment of inertia to a specific value

        Args:
            moment_inertia (tuple): A tuple containg the moment of inertia in
                the form :math:`(L_x,L_y,L_z)`

        """
        self.moment_inertia = moment_inertia


class Trimer(Molecule):
    """Defines a Trimer molecule for initialisation within a hoomd context

    This defines a molecule of three particles, shaped somewhat like Mickey
    Mouse. The central particle is of type `'A'` while the outer two
    particles are of type `'B'`. The type `'B'` particles, have a variable
    radius and are positioned at a specified distance from the central
    type `'A'` particle. The angle between the two type `'B'` particles,
    subtended by the type `'A'` particle is the other degree of freedom.

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
        self.particles = ["B", "B"]
        self.moment_inertia = (0, 0, 1.65)

    def define_potential(self):
        R"""Define the potential in the simulation context

        A helper function that defines the potential to be used by the  hoomd
        simulation context. The default values for the potential are a
        Lennard-Jones potential with a cutoff of 2.5 and interaction parameters
        of :math:`\epsilon = 1.0` and :math:`\sigma = 2.0`.

        Returns:
            class:`hoomd.md.pair`: The interaction potential object class.
        """
        potential = super(Trimer, self).define_potential()
        potential.pair_coeff.set('B', 'B', epsilon=1, sigma=self.radius*2)
        potential.pair_coeff.set('A', 'B', epsilon=1, sigma=1.0+self.radius)
        return potential

    def define_rigid(self, create=False, params=None):
        """Define the rigid constraints of the Trimer molecule

        This is a helper function to define the rigid body constraints of the
        particular molecules within the hoomd context.

        Args:
            create (bool): Flag that toggles the option of creating the
                additional particles when creating the rigid bodies. Defaults
                to False.
            params (dict): Dictionary defining the rigid body structure. The
                default values for the `type_name` of A and the `types` of the
                `self.particles` variable should work for the vast majority of
                systems, so the only value required should be the topology.

        Returns:
            class:`hoomd.md.constrain.rigid`: Rigid constraint object
        """
        angle = (self.angle/2)*math.pi/180.
        if not params:
            params = dict()
        params.setdefault('positions', [
            (math.sin(angle), math.cos(angle), 0),
            (-math.sin(angle), math.cos(angle), 0)
        ])
        rigid = super(Trimer, self).define_rigid(create, params)
        return rigid


class Dimer(Molecule):
    """Defines a Trimer molecule for initialisation within a hoomd context

    This defines a molecule of three particles, shaped somewhat like Mickey
    Mouse. The central particle is of type `'A'` while the outer two
    particles are of type `'B'`. The type `'B'` particles, have a variable
    radius and are positioned at a specified distance from the central
    type `'A'` particle. The angle between the two type `'B'` particles,
    subtended by the type `'A'` particle is the other degree of freedom.

    Args:
        radius (float): Radius of the small particles. Default is 0.637556
        distance (float): Distance of the outer particles from the central
            one. Default is 1.0
        angle (float): Angle between the two outer particles in degrees.
            Default is 120
    """
    def __init__(self, radius=0.637556, distance=1.0):
        super(Dimer, self).__init__()
        self.radius = radius
        self.distance = distance
        self.particles = ["B"]
        self.moment_inertia = self.compute_moment_intertia()

    def compute_moment_intertia(self):
        """Compute the moment of inertia from the particle paramters"""
        return (0, 0, 2*(self.distance/2)**2)

    def define_potential(self):
        R"""Define the potential in the simulation context

        A helper function that defines the potential to be used by the  hoomd
        simulation context. The default values for the potential are a
        Lennard-Jones potential with a cutoff of 2.5 and interaction parameters
        of :math:`\epsilon = 1.0` and :math:`\sigma = 2.0`.

        Returns:
            class:`hoomd.md.pair`: The interaction potential object class.
        """
        potential = super(Dimer, self).define_potential()
        potential.pair_coeff.set('B', 'B', epsilon=1, sigma=self.radius*2)
        potential.pair_coeff.set('A', 'B', epsilon=1, sigma=1.0+self.radius)
        return potential

    def define_rigid(self, create=False, params=None):
        """Define the rigid constraints of the molecule

        This is a helper function to define the rigid body constraints of the
        particular molecules within the hoomd context.

        Args:
            create (bool): Flag that toggles the option of creating the
                additional particles when creating the rigid bodies. Defaults
                to False.
            params (dict): Dictionary defining the rigid body structure. The
                default values for the `type_name` of A and the `types` of the
                `self.particles` variable should work for the vast majority of
                systems, so the only value required should be the topology.

        Returns:
            class:`hoomd.md.constrain.rigid`: Rigid constraint object
        """
        if not params:
            params = dict()
        params.setdefault('positions', [(self.distance, 0, 0)])
        rigid = super(Dimer, self).define_rigid(create, params)
        return rigid


if __name__ == "__main__":
    hoomd.context.initialize()
    hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4), n=[4, 4])
    Trimer().initialise()
