#!/usr/bin/env python
""" A set of classes used for computing the dynamic properties of a Hoomd MD
simulation"""

from __future__ import print_function
import numpy as np
import quaternion
from CompDynamics import CompDynamics, CompRotDynamics
import pandas


class TimeDep(object):
    """ Class to compute the time dependent characteristics of individual
    particles in a hoomd simulation.

    Args:
        snapshot (:class:`hoomd.data.SnapshotParticleData`): Hoomd snapshot
            object which is the initial configuration for the purposes of the
            dynamics calculations
    """
    def __init__(self, snapshot, timestep):
        self.t_init = snapshot
        self.timestep = timestep
        self._init_snapshot(snapshot)
        self._data = self.get_data(snapshot, timestep)

    def _init_snapshot(self, snapshot):
        self.pos_init = self.t_init.particles.position
        self.image_init = self.t_init.particles.image

    def get_time_diff(self, timestep):
        """ The difference in time between the currrent timestep and the
        timestep of the initial configuration.

        Args:
            snapshot (:class:`hoomd.data.SnapshotParticleData`): The snapshot
                the difference is to be calculated at

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
        try:
            box_dim = np.array([
                snapshot.box.Lx,
                snapshot.box.Ly,
                snapshot.box.Lz
            ])
        except AttributeError:
            box_dim = np.array([
                snapshot.configuration.box[0],
                snapshot.configuration.box[1],
                snapshot.configuration.box[2]
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


    def get_data(self, snapshot, timestep):
        """ Get translational and rotational data

        Args:
            system (:class:`hoomd.data.SnapshotParticleData`): Hoomd system
                object in the configuration to be saved

        Returns:
            :class:`TransData`: Data object

        """
        data = pandas.DataFrame({
            'displacement': self._displacement(snapshot),
            'time': self.get_time_diff(timestep)
        })
        data.time_diff = self.get_time_diff(timestep)
        return data

    def get_all_data(self):
        return self._data

    def print_all(self, snapshot, timestep, outfile):
        """Print all data to file

        Args:
            system (:class:`hoomd.data.SnapshotParticleData`): Hoomd snapshot
                object in the configuration to be saved
            outfile (string): filename to output data to
        """
        data = self.get_data(snapshot, timestep)
        CompDynamics(data).print_all(outfile)
        # data.to_json(outfile[:-8]+"-tr.dat")


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
        snapshot (:class:`hoomd.data.SnapshotParticleData`): Hoomd snapshot
            object at the initial time for the dyanamics computations

    """
    def __init__(self, snapshot, timestep):
        super(TimeDep2dRigid, self).__init__(snapshot, timestep)

    def _init_snapshot(self, snapshot):
        super()._init_snapshot(snapshot)
        self.bodies = np.max(self.t_init.particles.body)+1
        self.orient_init = self.array2quat(
            self.t_init.particles.orientation[:self.bodies])

    @staticmethod
    def array2quat(array):
        """Convert a numpy array to an array of quaternions"""
        return quaternion.as_quat_array(array.astype(float))

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
        orient_final = self.array2quat(
            snapshot.particles.orientation[:self.bodies])
        rot_q = orient_final/self.orient_init
        rot = quaternion.as_rotation_vector(rot_q).sum(axis=1)
        rot[rot > np.pi] -= 2*np.pi
        rot[rot < -np.pi] += 2*np.pi
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
        return np.sqrt(np.power(curr - self.pos_init[:self.bodies], 2).sum(1))

    def _all_displacement(self, snapshot):
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
        curr = self._unwrap(snapshot)
        return np.sqrt(np.power(curr - self.pos_init, 2).sum(1))

    def append(self, snapshot, timestep):
        self._data = self._data.append(self.get_data(snapshot, timestep))

    def get_data(self, snapshot, timestep):
        """ Get translational and rotational data

        Args:
            snapshot (:class:`hoomd.data.SnapshotParticleData`): Hoomd data
                object

        Returns:
        """
        data = pandas.DataFrame({
            'displacement': self._displacement(snapshot),
            'rotation': self._rotations(snapshot),
            'time': self.get_time_diff(timestep),
        })
        data.bodies = self.bodies
        return data

    def get_all_data(self):
        return self._data

    def print_all(self, snapshot, timestep, outfile):
        """Print all data to file

        Args:
            snapshot (:class:`hoomd.data.SnapshotParticleData`): Hoomd snapshot
                object in the configuration to be saved
            outfile (string): filename to output data to
        """
        data = self.get_data(snapshot, timestep)
        CompRotDynamics(data).print_all(outfile)
