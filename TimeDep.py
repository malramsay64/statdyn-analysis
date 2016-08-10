#!/usr/bin/env python
""" A set of classes used for computing the dynamic properties of a Hoomd MD
simulation"""

from __future__ import print_function
import math
import numpy as np
import quaternion
from TransData import TransData, TransRotData
from CompDynamics import CompDynamics, CompRotDynamics

class TimeDep(object):
    """ Class to compute the time dependent characteristics of individual
    particles in a hoomd simulation.

    Args:
        system (system): Hoomd system object which is the initial configuration
            for the purposes of the dynamics calculations
    """
    def __init__(self, system):
        t_init = self._take_snapshot(system)
        self.pos_init = self._unwrap(t_init)
        self.timestep = system.get_metadata()['timestep']

    def _take_snapshot(self, system):
        """ Takes a snapshot of the current configuration of the system

        Args:
            system (system): Hoomd system objet in the configuration to be saved

        Returns:
            Snapshot: An immutable snapshot of the system at the current time
        """
        return system.take_snapshot()

    def get_time_diff(self, timestep):
        """ The difference in time between the currrent timestep and the
        timestep of the initial configuration.

        Args:
            timestep (int): The timestep the difference is to be calculated at

        Returns:
            int: Difference between initial and current timestep
        """
        return timestep - self.timestep

    def _unwrap(self, snapshot):
        """ Unwraps periodic positions to absolute positions

        Converts the periodic potition of each particle to it's *real* position
        by taking into account the *image* the particle exitsts on.

        Args:
            snapshot (snapshot): Hoomd snapshot of the system to unwrap

        Returns:
            :class:`numpy.array`: A numpy array of position arrays contianing
            the unwrapped positions
        """
        box_dim = np.array([snapshot.box.Lx,
                            snapshot.box.Ly,
                            snapshot.box.Lz
                           ])
        pos = np.array(snapshot.particles.position)
        image = np.array(snapshot.particles.image)
        return pos + image*box_dim

    def _displacement(self, snapshot):
        """ Calculate the squared displacement for all bodies in the system.

        This is the function that computes the per body displacements for all
        the dynamic quantities. This is where the most computation takes place
        and can be called once for a number of further calculations.

        Args:
            snapshot (snapshot): The configuration at which the difference is
                computed

        Returns:
            :class:`numpy.array`: An array containing per body displacements
        """
        curr = self._unwrap(snapshot)
        return np.sqrt(np.power(curr - self.pos_init, 2).sum(1))

    def get_data(self, system):
        """ Get translational and rotational data """

        return TransData().from_trans_array(
            self._displacement(self._take_snapshot(system)),
            self.get_time_diff(system.get_metadata['timestep']))

    def print_all(self, system, outfile):
        """Print all data to file"""
        data = self.get_data(system)
        CompDynamics(data).print_all(outfile)
        data.to_json(outfile+".json")


class TimeDep2dRigid(TimeDep):
    """ Class to compute the time dependent characteristics of 2D rigid bodies
    in a hoomd simulation.

    This class extends on from :class:TimeDep computing rotational properties of
    the 2D rigid bodies in the system. The distinction of two dimensional
    molecules makes for a simpler analysis of the rotational characteristics.

    Note:
        This class computes positional properties for each rigid body in the
        system. This means that for computations of rigid bodies it would be
        expected to get different translational quantities using the
        :class:TimeDep and the :class:TimeDep2dRigid classes.

    Warning:
        Hoomd doesn't store rotational data in a regular snapshot, even
        with the ``rigid_bodies=True`` option. My solution has been to
        modify the source code to add this functionality to taking
        snapshots.

    Todo:
        Have the capability to get the orientations from the system when taking
        the snapshot allowing for the computation of rotation on a default
        hoomd install.

    Args:
        system (system): Hoomd system object at the initial time for the
            dyanamics computations

    """
    def __init__(self, system):
        super(TimeDep2dRigid, self).__init__(system)
        t_init = self._take_snapshot(system)
        self.orient_init = np.array([scalar4_to_quaternion(i)
                                     for i in t_init.bodies.orientation])

    def _unwrap(self, snapshot):
        """ Unwraps the periodic positions to absolute positions

        Converts the periodic positions of each rigid body to absolute
        positions for computation of displacements.

        Note:
            This function overwrites the base class function due to differences
            in the way the rigid and non-rigid positions are stored in a
            snapshot

        Args:
            snapshot (snapshot): Hoomd snapshot of the configuration to unwrap

        Return:
            :class:`numpy.array`: A Numpy array of position arrays for each
            rigid body center of mass.
        """
        box_dim = np.array([snapshot.box.Lx,
                            snapshot.box.Ly,
                            snapshot.box.Lz
                           ])
        pos = np.array([scalar3_to_array(i) for i in snapshot.bodies.com])
        image = np.array([scalar3_to_array(i)
                          for i in snapshot.bodies.body_image])
        return pos + image*box_dim

    def _take_snapshot(self, system):
        """ Takes a snapshot of the system including rigid bodies

        Note:
            This overrides the :func:`_take_snapshot` of the base
            :class:`TimeDep` class to specify that that snapshot includes the
            rigid body data.

        Args:
            system (system): The hoomd system object

        Return:
            snapshot (snapshot): The snapshot of the system including the rigid
                body data
        """
        return system.take_snapshot(rigid_bodies=True)

    def _rotations(self, snapshot):
        R""" Calculate the rotation for every rigid body in the system

        This calculates the angle rotated between the initial configuration and
        the current configuration. It doesn't take into accout multiple
        rotations with values falling in the range :math:`[\-pi,\pi)`.

        Args:
            snapshot (snapshot): The final configuration

        Return:
            :class:`numpy.array`: Array of all the rotations
        """
        orient_final = np.array([scalar4_to_quaternion(i)
                                 for i in snapshot.bodies.orientation])
        rot_q = orient_final/self.orient_init
        rot = quaternion.as_rotation_vector(rot_q)[:, 0]
        for i, val in enumerate(rot):
            if val > math.pi:
                rot[i] -= 2*math.pi
            elif val < -math.pi:
                rot[i] += 2*math.pi
        return rot

    def get_data(self, system):
        """ Get translational and rotational data """
        snap = self._take_snapshot(system)

        return TransRotData().from_arrays(
            self._displacement(snap),
            self._rotations(snap),
            self.get_time_diff(system.get_metadata['timestep']))

    def print_all(self, system, outfile):
        """Print all data to file"""
        data = self.get_data(system)
        CompRotDynamics(data).print_all(outfile)
        data.to_json(outfile+".json")

def scalar3_to_array(scalar):
    """ Convert scalar3 representation to a numpy array

    Args:
        scalar (scalar3): Scalar3 position array

    Return:
        :class:`numpy.array`: Position array
    """
    return np.array([scalar.x, scalar.y, scalar.z])

def scalar4_to_array(scalar):
    """ Convert scalar4 representation to a numpy array

    Args:
        scalar (scalar4): Scalar4 position array

    Return:
        :class:`numpy.array`: numpy array
    """
    return np.array([scalar.x, scalar.y, scalar.z, scalar.w])

def scalar4_to_quaternion(scalar):
    R""" Convert scalar4 representation to a quaternion

    Conversion to quaternion where the *w* component of the scalar
    is the real part of the quaternion and the *x,y,z* values
    are the vector part.

    Args:
        scalar (scalar4): Scalar4 object representing a quaternion

    Return:
        :class:`quaternion.quaternion`: quaternion object
    """
    return quaternion.quaternion(scalar.w, scalar.x, scalar.y, scalar.z)
