#!/usr/bin/env python
""" A set of classes used for computing the dynamic properties of a Hoomd MD
simulation"""

from __future__ import print_function
import os
import math
import numpy as np
from scipy import stats
from hoomd_script import (init,
                          update,
                          pair,
                          integrate,
                          analyze,
                          group,
                          run_upto,
                          run,
                          dump)
import StepSize

class TimeDep(object):
    """ Class to compute the time dependent characteristics of individual
    particles in a hoomd simulation.

    Args:
        system (system): Hoomd system object which is the initial configuration
            for the purposes of the dynamics calculations
    """
    def __init__(self, system):
        self.t_init = self._take_snapshot(system)
        self.pos_init = self._unwrap(self.t_init)
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

    def _displacement_sq(self, snapshot):
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
        return np.power(curr - self.pos_init, 2).sum(axis=1)

    def _calc_mean_disp(self, displacement_sq):
        """ Performs the calculation of the mean displacement

        Args:
            displacement_sq (:class:`numpy.array`): Array of squared
                displacements

        Return:
            float: The mean displacement
        """
        return np.mean(np.sqrt(displacement_sq))

    def get_mean_disp(self, system):
        R""" Compute the mean displacement

        Finds the mean displacement of the current configuration from the
        initial configuration upon initialising the class.

        .. math::
            \langle \Delta r \rangle = \langle \sqrt{x^2 + y^2 + z^2} \rangle

        Args:
            system (system): Hoomd system object at currrent time

        Return:
            float: The mean displacement
        """
        return self._calc_mean_disp(
            self._displacement_sq(self._take_snapshot(system)))

    def _calc_msd(self, displacement_sq):
        """ Performs the calculation of the mean squared displacement

        Args:
            displacement_sq (:class:`numpy.array`): Array of squared
            displacements

        Return:
            float: The mean of the squared displacements
        """
        return np.mean(displacement_sq)

    def get_msd(self, system):
        R""" Compute the mean squared displacement

        Finds the mean squared displacement of the current state from the inital
        state of the initialisaiton of the class.

        .. math:: MSD = \langle \Delta r^2 \rangle

        Args:
            system (system): Hoomd system object at the current time

        Return:
            float: The mean squared displacement
        """
        return self._calc_msd(
            self._displacement_sq(self._take_snapshot(system)))

    def _calc_mfd(self, displacement_sq):
        """ Performs the calculation of the mean fourth displacement

        Args:
            displacement_sq (:class:`numpy.array`): Array of squared
            displacements

        Return:
            float: The mean of the displacements to the fourth power
        """
        return np.mean(np.power(displacement_sq, 2))

    def get_mfd(self, system):
        R""" Compute the mean fourth disaplacement

        Finds the mean of the displacements to the fourth power from
        the initial state to the current configuration.

        .. math:: MFD = \langle \Delta r^4 \rangle

        Args:
            system (system): Hoomd system object at the current time

        Return:
            float: The mean fourth displacement
        """
        return self._calc_mfd(
            self._displacement_sq(self._take_snapshot(system)))

    def _calc_alpha(self, displacement_sq):
        """ Performs the calculation of the non-gaussian parameter

        Args:
            displacement_sq (:class:`numpy.array`): Array of squared
            displacements

        Return:
            float: The non-gaussian parameter alpha
        """
        msd = self._calc_msd(displacement_sq)
        mfd = self._calc_mfd(displacement_sq)
        return mfd/(2.*(msd*msd)) - 1

    def get_alpha(self, system):
        R""" Compute the non-gaussian parameter :math:`\alpha`

        The non-gaussian parameter is given as

        .. math::
            \alpha = \frac{\langle \Delta r^4\rangle}
                      {\langle \Delta r^2  \rangle^2} -1

        Args:
            system (system): Hoomd system object at the current time

        Return:
            float: The non-gaussian parameter :math:`\alpha`
        """
        return self._calc_alpha(
            self._displacement_sq(self._take_snapshot(system)))


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
        rot = np.empty(len(self.t_init.bodies.com))
        for i in range(len(self.t_init.bodies.com)):
            rot[i] = (quat_to_2d(self.t_init.bodies.orientation[i])
                      - quat_to_2d(snapshot.bodies.orientation[i]))
            if rot[i] > math.pi:
                rot[i] = 2*math.pi - rot[i]
        return rot


    def _calc_decoupling(self, displacement_sq, rotations,
                         delta_disp, delta_rot):
        """ Calculates the decoupling of rotations and translations.

        Args:
            snapshot (snapshot): The snapshot of the current configuration
            delta_disp (float): The bin size of the displacement for integration
            delta_rot (float):  The bin size of the rotations for integration

        Return:
            float: The decoupling parameter
        """
        # Calculate and bin displacements
        disp = np.sqrt(displacement_sq)
        disp = np.floor(disp/delta_disp).astype(int)
        # adding 1 to account for 0 value
        disp_max = np.max(disp+1)
        disp_array = np.asmatrix(np.power(
            np.arange(1, disp_max+1)*delta_disp, 2))
        # Calculate and bin rotaitons
        rot = np.floor(np.abs(rotations)/delta_rot).astype(int)
        # adding 1 to account for 0 value
        rot_max = np.max(rot+1)
        rot_array = np.asmatrix(np.sin(
            np.arange(1, rot_max+1)*delta_rot))
        # Use binned values to create a probability matrix
        prob = np.zeros((rot_max, disp_max))
        for i, j in zip(rot, disp):
            prob[i][j] += 1

        prob = normalise_probability(np.asmatrix(prob),
                                     rot_array, disp_array,
                                     delta_rot, delta_disp
                                    )


        # Calculate tranlational and rotational probabilities
        p_trans = (prob.transpose() * rot_array.transpose())
        p_trans *= delta_rot
        p_rot = (prob * disp_array.transpose())
        p_rot *= delta_disp

        # Calculate the squared difference between the combined and individual
        # probabilities and then integrate over the differences to find the
        # coupling strength
        diff2 = np.power(prob - p_rot * p_trans.transpose(), 2)
        decoupling = ((diff2 * np.power(disp_array, 2).transpose())
                      * np.power(rot_array, 2)).sum()
        decoupling /= ((prob*disp_array.transpose()) * rot_array).sum()
        return decoupling.sum()

    def get_decoupling(self, system, delta_disp=0.005, delta_rot=0.005):
        """ Compute the decoupling of the translational and rotational motion

        This computs the coupling strength parameter as described by
        Farone and Chen

        References:
            A. Farone, L. Liu, S.-H. Chen, J. Chem. Phys. 119, 6302 (2003)

        Args:
            system (system): Hoomd system at the currrent time
            delta_disp (float): The bin size of displacements for integration
            delta_rot (float): The bin size of rotations for integration

        Return:
            float: The decoupling parameter
        """
        snapshot = self._take_snapshot(system)
        return self._calc_decoupling(self._displacement_sq(snapshot),
                                     self._rotations(snapshot),
                                     delta_disp,
                                     delta_rot
                                    )

    def _calc_mean_rot(self, rotations):
        """ Calculate the mean rotation given all the rotations

        This doesn't take into account the direction of the rotation,
        only its magnitude.

        Arg:
            rotations (:class:`numpy.array`): Array of all rotations

        Retrun:
            float: The mean rotation
        """
        return np.mean(np.abs(rotations))

    def get_mean_rot(self, system):
        R""" Compute the mean rotational distance

        This finds the mean rotational magnitude, taking the absolute value
        of the rotations and ignoring their directions.

        .. math:: \langle |\Delta \theta |\rangle

        Args:
            system (system): The system at the end of the motion

        Return:
            float: The mean rotation in radians
        """
        return self._calc_mean_rot(
            self._rotations(self._take_snapshot(system)))

    def _calc_mean_sq_rot(self, rotations):
        """ Calculate the mean squared rotation

        Args:
            rotations (:class:`numpy.array`): Array containing the rotations of
                each rigid body

        Return:
            float: The mean squared rotation
        """
        return np.mean(np.power(rotations, 2))

    def get_mean_sq_rot(self, system):
        R""" Compute the mean squared rotation

        Computes the mean of the squared rotation

        .. math:: MSR = \langle \Delta \theta^2 \rangle

        Args:
            system (system): System at the end of the motion

        Return:
            float: The mean squared rotation in radians
        """
        return self._calc_mean_sq_rot(
            self._rotations(self._take_snapshot(system)))

    def _calc_mean_trans_rot(self, disp_sq, rotations):
        """ Calculate the coupled translation and rotation

        Args:
            disp_sq (:class:`numpy.array`): Array containing the squared
                displacements
            rotations (class:`numpy.array`): Array of the rotations

        Return:
            float: The coupling of translations and rotations
        """
        return np.mean(np.sqrt(disp_sq) * np.abs(rotations))

    def get_mean_trans_rot(self, system):
        R""" Compute the coupled translation and rotation

        A measure of the coupling of translations and rotations

        .. math:: \langle \Delta r \Delta \theta \rangle

        Args:
            system (system): The hoomd system objet

        Return:
            float: The coupling parameter
        """
        snapshot = self._take_snapshot(system)
        return self._calc_mean_trans_rot(self._displacement_sq(snapshot),
                                         self._rotations(snapshot)
                                        )

    def _calc_mean_sq_trans_rot(self, disp_sq, rotations):
        R""" Calculate the squared coupled translation and rotation

        Args:
            disp_sq (:class:`numpy.array`): Array containing the squared
                displacements
            rotations (class:`numpy.array`): Array of the rotations

        Return:
            float: The squared coupling of translations and rotations
        """
        return np.mean(disp_sq * np.power(rotations, 2))

    def get_mean_sq_trans_rot(self, system):
        R""" Return the squared coupled translation and rotation

        A measure of the coupling of translations and rotations

        .. math:: \langle \Delta r^2 \Delta \theta^2 \rangle

        Args:
            system (system): The hoomd system object

        Return:
            float: The squared coupling of translations and rotations
        """
        snapshot = self._take_snapshot(system)
        return self._calc_mean_sq_trans_rot(self._displacement_sq(snapshot),
                                            self._rotations(snapshot)
                                           )

    def _calc_gamma1(self, disp_sq, rotations):
        R""" Calculate the first order coupling of translations and rotations

        Args:
            disp_sq (:class:`numpy.array`): Array containing the squared
                displacements
            rotations (class:`numpy.array`): Array of the rotations

        Return:
            float: The coupling of translations and rotations :math:`\gamma_1`
        """
        return ((self._calc_mean_trans_rot(disp_sq, rotations)
                 - self._calc_mean_disp(disp_sq)*self._calc_mean_rot(rotations))
                / np.sqrt(self._calc_msd(disp_sq)*
                          self._calc_mean_sq_rot(rotations)))

    def get_gamma1(self, system):
        R""" Calculate the first order coupling of translations and rotations

        .. math::
            \gamma_1 &= \frac{\langle\Delta r |\Delta\theta| \rangle -
                \langle\Delta r\rangle\langle| \Delta \theta |\rangle }
                {\sqrt{\langle\Delta r^2\rangle\langle\Delta\theta^2\rangle}}

        Args:
            system (system): The hoomd sytem object

        Return:
            float: The :math:`\gamma_1` value
        """
        snapshot = self._take_snapshot(system)
        return self._calc_gamma1(self._displacement_sq(snapshot),
                                 self._rotations(snapshot))


    def _calc_gamma2(self, disp_sq, rotations):
        R""" Calculate the second order coupling of translations and rotations

        Args:
            disp_sq (:class:`numpy.array`): Array containing the squared
                displacements
            rotations (class:`numpy.array`): Array of the rotations

        Return:
            float: The squared coupling of translations and rotations
            :math:`\gamma_2`
        """
        return ((self._calc_mean_sq_trans_rot(disp_sq, rotations)
                 - self._calc_msd(disp_sq)*self._calc_mean_sq_rot(rotations))
                / (self._calc_msd(disp_sq)
                   * self._calc_mean_sq_rot(rotations)))

    def get_gamma2(self, snapshot):
        R""" Calculate the second order coupling of translations and rotations

        .. math:: \gamma_2 &= \frac{\langle(\Delta r \Delta\theta)^2 \rangle -
                \langle\Delta r^2\rangle\langle\Delta \theta^2\rangle
                }{\langle\Delta r^2\rangle\langle\Delta\theta^2\rangle}

        Args:
            system (system): The hoomd sytem object

        Return:
            float: The squared coupling of translations and rotations
            :math:`\gamma_2`
        """
        return self._calc_gamma2(self._displacement_sq(snapshot),
                                 self._rotations(snapshot))

    def _calc_corr_dist(self, disp_sq, rotations):
        R"""Calculate the correlation of residuals for the translations and
        rotations

        .. math:: Q =  (\Delta \theta - \langle \Delta \theta \rangle)
                (\Delta r - \langle \Delta r \rangle)

        Args:
            disp_sq (:class:`numpy.array`): Array containing the squared
                displacements
            rotations (class:`numpy.array`): Array of the rotations

        Return:
            :class:`numpy.array`: Array of corrrelated residuals
        """
        return ((np.sqrt(disp_sq) - self._calc_msd(disp_sq))
                * (rotations - self._calc_mean_rot(rotations)))

    def print_corr_dist(self, system, outfile='dist.dat'):
        R"""Print all corrlation values to a file

        .. math:: Q = (\Delta \theta - \langle \Delta \theta \rangle)
                (\Delta r - \langle \Delta r \rangle)

        Args:
            system (:class:`hoomd.system`): Hoomd system object
            outfile (string): Filename to append output to
        """
        snapshot = self._take_snapshot(system)
        timestep = system.get_metadata()['timestep']
        distribution = self._calc_corr_dist(self._displacement_sq(snapshot),
                                            self._rotations(snapshot))
        for val in distribution:
            print(self.get_time_diff(timestep), val, file=open(outfile, 'a'))

    def _calc_corr_skew(self, disp_sq, rotations):
        """Compute the skew of the distribution of the correlation of
        translations and rotaions

        Args:
            disp_sq (:class:`numpy.array`): Array containing the squared
            displacements
            rotations (class:`numpy.array`): Array of the rotations

        Return:
            float: The skew of the correlation distribution
        """
        return stats.skew(self._calc_corr_dist(disp_sq, rotations))

    def get_corr_skew(self, system):
        """Compute the skew of the distribution of the correlation of
        translations and rotaions

        Args:
            system (system): The hoomd sytem object

        Return:
            float: The skew of the correlation distribution
        """
        snapshot = self._take_snapshot(system)
        return self._calc_corr_skew(self._displacement_sq(snapshot),
                                    self._rotations(snapshot))

    def print_all(self, system, outfile=None):
        """ Print all dynamic quantities to a file

        Prints all the calculated dynamic quantities to either
        stdout or a file. This function only calculates the distances and
        rotations a single time using the private calc methods.

        Args:
            system (system): The hoomd sytem object
            outfile (string): Filename to append to
        """
        snapshot = self._take_snapshot(system)
        timestep = system.get_metadata()['timestep']
        disp_sq = self._displacement_sq(snapshot)
        rotations = self._rotations(snapshot)
        output = dict()
        output['msd'] = self._calc_msd(disp_sq)
        output['mfd'] = self._calc_mfd(disp_sq)
        output['alpha'] = self._calc_alpha(disp_sq)
        output['disp'] = self._calc_mean_disp(disp_sq)
        output['mean_rot'] = self._calc_mean_rot(rotations)
        output['time'] = self.get_time_diff(timestep)
        output['decoupling'] = self._calc_decoupling(
            disp_sq, rotations, 0.05, 0.05)
        output['gamma1'] = self._calc_gamma1(disp_sq, rotations)
        output['gamma2'] = self._calc_gamma2(disp_sq, rotations)
        output['correlation'] = self._calc_corr_skew(disp_sq, rotations)
        if outfile:
            print(vals_to_string(output), file=open(outfile, 'a'))
        else:
            print(vals_to_string(output))


    def print_heading(self, outfile):
        """ Write heading values to outfile which match up with the values given
        by print_all().

        Args:
            outfile (string): Filename to write headings to
        """
        output = dict()
        output['msd'] = 0
        output['mfd'] = 0
        output['alpha'] = 0
        output['disp'] = 0
        output['mean_rot'] = 0
        output['time'] = 0
        output['decoupling'] = 0
        output['gamma1'] = 0
        output['gamma2'] = 0
        output['correlation'] = 0
        print(keys_to_string(output), file=open(outfile, 'w'))

def keys_to_string(dictionary, sep=' '):
    """Converts all keys in a dictionary to a string

    Args:
        dictionary (dict): Dictionary of key, value pairs
        sep (string): Separator for between keys

    Return:
        string: All keys separated by `sep`
    """
    return sep.join([str(key) for key in dictionary.keys()])

def vals_to_string(dictionary, sep=' '):
    """Converts all vals in a dictionary to a string

    Args:
        dictionary (dict): Dictionary of key, value pairs
        sep (string): Separator for between values

    Return:
        string: All values separated by `sep`
    """
    return sep.join([str(val) for val in dictionary.values()])

def quat_to_2d(quat):
    """ Convert quaternion to angle in 2D plane

    Convert the quaternion representation of angle to a two dimensional
    angle in the xy plane.

    Args:
        quat (scalar4): Quaternion representation of an angle

    Return:
        float: Angle rotated on the xy plane
    """
    return math.atan2(quat.x*quat.w + quat.y*quat.z,
                      0.5-quat.y*quat.y - quat.z-quat.z)

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


def normalise_probability(prob_matrix, rot_matrix, disp_matrix,
                          delta_rot=0.005, delta_disp=0.005):
    """ Function to normalise the probabilty matrix of the decoupling parameter
    to integrate to a value of 1.

    The components are normalised through the use of matrix multiplication

    Args:
        prob_matrix (:class:`numpy.matrix`): The numpy matrix to be normalised
        rot_matrix (:class:`numpy.matrix`): The numpy matrix of theta values
            these include any transformations that are involved in the
            integration
        disp_matrix (:class:`numpy.matrix`): The numpy matrix of displacement
            values with any transformations already applied.
        delta_disp (float): The distance between displacements (i.e.
            the binning) delta_rot
        delta_rot (float): The distance between rotations (i.e. the binning)

    Return:
        :class:`numpy.matrix`: Normalised matrix
    """
    factor = (((rot_matrix * prob_matrix) * disp_matrix.transpose()) *
              delta_disp * delta_rot)
    return prob_matrix/factor

def compute_dynamics(input_xml,
                     temp,
                     press,
                     steps,
                     rigid=True):
    """ Run a hoomd simulation calculating the dynamic quantites on a power
    law scale such that both short timescale and long timescale events are
    vieable on the same figure while retaining a reasonable runtime.
    for the simulation

    Args:
        input_xml (string): Filename of the file containing the input
            configuration
        temp (float): The target temperature at which to run the simulation
        press (float): The target pressure at which to run the simulation
        rigid (bool): Boolean value indicating whether to integrate using rigid
            bodes.
    """
    if init.is_initialized():
        init.reset()
    basename = os.path.splitext(input_xml)[0]

    # Fix for issue where pressure is higher than desired
    press /= 2.2

    # Initialise simulation parameters
    # context.initialize()
    system = init.read_xml(filename=input_xml, time_step=0)
    update.enforce2d()

    potentials = pair.lj(r_cut=2.5)
    potentials.pair_coeff.set('1', '1', epsilon=1, sigma=2)
    potentials.pair_coeff.set('2', '2', epsilon=1, sigma=0.637556*2)
    potentials.pair_coeff.set('1', '2', epsilon=1, sigma=1.637556)

    # Create particle groups
    gall = group.all()

    # Set integration parameters
    integrate.mode_standard(dt=0.005)
    if rigid:
        integrate.npt_rigid(group=gall, T=temp, tau=1, P=press, tauP=1)
    else:
        integrate.npt(group=gall, T=temp, tau=1, P=press, tauP=1)

    # initial run to settle system after reading file
    run(10000)

    thermo = analyze.log(filename=basename+"-thermo.dat",
                         quantities=['temperature', 'pressure',
                                     'potential_energy',
                                     'rotational_kinetic_energy',
                                     'translational_kinetic_energy'
                                    ],
                         period=1000)

    # Initialise dynamics quantities
    dyn = TimeDep2dRigid(system)
    dyn.print_heading(basename+"-dyn.dat")
    tstep_init = system.get_metadata()['timestep']
    new_step = StepSize.PowerSteps(start=tstep_init)
    struct = [(new_step.next(), new_step, dyn)]
    timestep = tstep_init
    key_rate = 33000
    xml = dump.xml(all=True)
    xml.write(filename=input_xml)

    while timestep < steps+tstep_init:
        index_min = struct.index(min(struct))
        next_step, step_iter, dyn = struct[index_min]
        timestep = next_step
        print(timestep, file=open("timesteps.dat", 'a'))
        run_upto(timestep)
        dyn.print_all(system, outfile=basename+"-dyn.dat")
        # dyn.print_corr_dist(system, outfile=basename+"-corr.dat")

        struct[index_min] = (step_iter.next(), step_iter, dyn)
        # Add new key frame when a run reaches 10000 steps
        if (timestep % key_rate == 0 and
                len(struct) < 2000 and
                len([s for s in struct if s[0] == timestep+1]) == 0):
            new_step = StepSize.PowerSteps(start=timestep)
            struct.append((new_step.next(), new_step, TimeDep2dRigid(system)))
        xml.write(filename=input_xml)
    thermo.query('pressure')


