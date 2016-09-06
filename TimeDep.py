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
        system (:class:`hoomd.data.system_data`): Hoomd system object which is
            the initial configuration for the purposes of the dynamics
            calculations
    """
    def __init__(self, system):
        self.t_init = self._take_snapshot(system)
        self.pos_init = self.t_init.particles.position
        self.image_init = self.t_init.particles.image
        self.timestep = system.get_metadata()['timestep']

    def _take_snapshot(self, system):
        """ Takes a snapshot of the current configuration of the system

        Args:
            system (:class:`hoomd.data.system_data`): Hoomd system objet
                in the configuration to be saved

        Returns:
            :class:`hoomd.data.SnapshotParticleData`: An immutable snapshot
                of the system at the current time
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
        pos = snapshot.particles.position
        image = snapshot.particles.image - self.image_init
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
        """ Get translational and rotational data

        Args:
            system (:class:`hoomd.data.system_data`): Hoomd system objet in
                the configuration to be saved

        Returns:
            :class:`TransData`: Data object

        """
        data = TransData()
        data.from_trans_array(
            self._displacement(self._take_snapshot(system)),
            self.get_time_diff(system.get_metadata()['timestep']))
        return data

    def print_all(self, system, outfile):
        """Print all data to file

        Args:
            system (:class:`hoomd.data.system_data`): Hoomd system objet in the
                configuration to be saved
            outfile (string): filename to output data to
        """
        data = self.get_data(system)
        CompDynamics(data).print_all(outfile)
        data.to_json(outfile[:-8]+"-tr.dat")


class TimeDep2dRigid(TimeDep):
    """ Class to compute the time dependent characteristics of 2D rigid bodies
    in a hoomd simulation.

    This class extends on from :class:`TimeDep` computing rotational properties
    of the 2D rigid bodies in the system. The distinction of two dimensional
    molecules makes for a simpler analysis of the rotational characteristics.

    Note:
        This class computes positional properties for each rigid body in the
        system. This means that for computations of rigid bodies it would be
        expected to get different translational quantities using the
        :class:`TimeDep` and the :class:`TimeDep2dRigid` classes.

    Args:
        system (:class:`hoomd.data.system_data`): Hoomd system object at the
            initial time for the dyanamics computations

    """
    def __init__(self, system):
        super(TimeDep2dRigid, self).__init__(system)
        self.bodies = np.max(self.t_init.particles.body)+1
        self.pos_init = self.pos_init[:self.bodies]
        self.orient_init = quaternion.as_quat_array(np.array(
            self.t_init.particles.orientation[:self.bodies], dtype=float))

    def _rotations(self, snapshot):
        R""" Calculate the rotation for every rigid body in the system

        This calculates the angle rotated between the initial configuration and
        the current configuration. It doesn't take into accout multiple
        rotations with values falling in the range :math:`[\-pi,\pi)`.

        Args:
            snapshot (:class:`hoomd.data.SnapshotParticleData`): The final
                configuration

        Return:
            :class:`numpy.array`: Array of all the rotations
        """
        orient_final = quaternion.as_quat_array(np.array(
            snapshot.particles.orientation[:self.bodies], dtype=float))
        rot_q = orient_final/self.orient_init
        rot = quaternion.as_rotation_vector(rot_q).sum(axis=1)
        for i, val in enumerate(rot):
            if val > math.pi:
                rot[i] -= 2*math.pi
            elif val < -math.pi:
                rot[i] += 2*math.pi
        return rot

    def _displacement(self, snapshot):
        """ Calculate the squared displacement for all bodies in the system.

        This is the function that computes the per body displacements for all
        the dynamic quantities. This is where the most computation takes place
        and can be called once for a number of further calculations.

        Args:
            snapshot (:class:'hoomd.data.SnapshotParticleData'): The
                configuration at which the difference is computed

        Returns:
            :class:`numpy.array`: An array containing per body displacements
        """
        curr = self._unwrap(snapshot)[:self.bodies]
        return np.sqrt(np.power(curr - self.pos_init, 2).sum(1))

    def get_data(self, system):
        """ Get translational and rotational data

        Args:
            system (:class:`hoomd.data.system_data`): Hoomd data object

        Returns:
            :class:`TransRotData`: Translational and rotational data
        """
        snap = self._take_snapshot(system)

        data = TransRotData()
        data.from_arrays(
            self._displacement(snap),
            self._rotations(snap),
            self.get_time_diff(system.get_metadata()['timestep']))
        assert (issubclass(type(data), TransRotData)), type(data)
        return data

    def print_all(self, system, outfile):
        """Print all data to file

        Args:
            system (:class:`hoomd.data.system_data`): Hoomd system objet in
                the configuration to be saved
            outfile (string): filename to output data to
        """
        data = self.get_data(system)
        CompRotDynamics(data).print_all(outfile)
        data.to_json(outfile[:-8]+"-tr.dat")
