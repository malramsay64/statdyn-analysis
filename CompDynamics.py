#!/usr/bin/env python
""" A set of classes used for computing the dynamic properties of a Hoomd MD
simulation"""

from __future__ import print_function
import numpy as np
from TransData import TransData, TransRotData


class CompDynamics(object):
    """ Class to compute the time dependent characteristics of individual
    particles in a hoomd simulation.

    Args:
        system (system): Hoomd system object which is the initial configuration
            for the purposes of the dynamics calculations
    """
    def __init__(self, TData=TransData()):
        self.data = TData
        assert isinstance(TransData, self.data)

    def timestep(self):
        """Return the timestep difference"""
        return self.data.timesteps

    def translations(self):
        """Return the translation of each molecule"""
        return self.data.trans

    def get_mean_disp(self):
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
        return np.mean(self.translations())

    def get_msd(self):
        R""" Compute the mean squared displacement

        Finds the mean squared displacement of the current state from the inital
        state of the initialisaiton of the class.

        .. math:: MSD = \langle \Delta r^2 \rangle

        Args:
            system (system): Hoomd system object at the current time

        Return:
            float: The mean squared displacement
        """
        return np.mean(np.power(self.translations(), 2))

    def get_mfd(self):
        R""" Compute the mean fourth disaplacement

        Finds the mean of the displacements to the fourth power from
        the initial state to the current configuration.

        .. math:: MFD = \langle \Delta r^4 \rangle

        Return:
            float: The mean fourth displacement
        """
        return np.mean(np.power(self.translations(), 4))


    def get_alpha(self):
        R""" Compute the non-gaussian parameter :math:`\alpha`

        The non-gaussian parameter is given as

        .. math::
            \alpha = \frac{\langle \Delta r^4\rangle}
                      {\langle \Delta r^2  \rangle^2} -1

        Return:
            float: The non-gaussian parameter :math:`\alpha`
        """
        return self.get_mfd()/(np.power(self.get_msd(), 2)) - 1

    def print_all(self, outfile=None):
        """ Print all dynamic quantities to a file

        Prints all the calculated dynamic quantities to either
        stdout or a file. This function only calculates the distances and
        rotations a single time using the private calc methods.

        Args:
            outfile (string): Filename to append to
        """
        output = dict()
        output['msd'] = self.get_msd()
        output['mfd'] = self.get_mfd()
        output['alpha'] = self.get_alpha()
        output['disp'] = self.get_mean_disp()
        output['time'] = self.timestep()
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
        output['time'] = 0
        print(keys_to_string(output), file=open(outfile, 'w'))

class CompRotDynamics(CompDynamics):
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
    def __init__(self, RigidData=TransRotData()):
        super(CompRotDynamics, self).__init__(RigidData)
        assert isinstance(TransRotData, self.data)

    def rotations(self):
        R""" Calculate the rotation for every rigid body in the system

        This calculates the angle rotated between the initial configuration and
        the current configuration. It doesn't take into accout multiple
        rotations with values falling in the range :math:`[\-pi,\pi)`.

        Args:
            snapshot (snapshot): The final configuration

        Return:
            :class:`numpy.array`: Array of all the rotations
        """
        return self.data.rotations


    def get_decoupling(self, delta_disp=0.005, delta_rot=0.005):
        """ Calculates the decoupling of rotations and translations.

        References:
            A. Farone, L. Liu, S.-H. Chen, J. Chem. Phys. 119, 6302 (2003)

        Args:
            snapshot (snapshot): The snapshot of the current configuration
            delta_disp (float): The bin size of the displacement for integration
            delta_rot (float):  The bin size of the rotations for integration

        Return:
            float: The decoupling parameter
        """
        # Calculate and bin displacements
        disp = self.translations()
        disp = np.floor(disp/delta_disp).astype(int)
        # adding 1 to account for 0 value
        disp_max = np.max(disp+1)
        disp_array = np.asmatrix(np.power(
            np.arange(1, disp_max+1)*delta_disp, 2))
        # Calculate and bin rotaitons
        rot = np.floor(np.abs(self.rotations())/delta_rot).astype(int)
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

    def get_mean_rot(self):
        R""" Calculate the mean rotation given all the rotations

        This doesn't take into account the direction of the rotation,
        only its magnitude.

        .. math:: \langle |\Delta \theta |\rangle

        Arg:
            rotations (:class:`numpy.array`): Array of all rotations

        Retrun:
            float: The mean rotation
        """
        return np.mean(np.abs(self.rotations()))

    def get_mean_sq_rot(self):
        R""" Compute the mean squared rotation

        Computes the mean of the squared rotation

        .. math:: MSR = \langle \Delta \theta^2 \rangle

        Args:
            system (system): System at the end of the motion

        Return:
            float: The mean squared rotation in radians
        """
        return np.mean(np.power(self.rotations(), 2))

    def get_mean_trans_rot(self):
        R""" Compute the coupled translation and rotation

        A measure of the coupling of translations and rotations

        .. math:: \langle \Delta r \Delta \theta \rangle

        Args:
            system (system): The hoomd system objet

        Return:
            float: The coupling parameter
        """
        return np.mean(self.translations()*np.abs(self.rotations()))

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

    def get_mean_sq_trans_rot(self):
        R""" Return the squared coupled translation and rotation

        A measure of the coupling of translations and rotations

        .. math:: \langle \Delta r^2 \Delta \theta^2 \rangle

        Args:
            system (system): The hoomd system object

        Return:
            float: The squared coupling of translations and rotations
        """
        return np.mean(np.power(self.translations(), 2)
                       * np.power(self.rotations(), 2))

    def get_gamma1(self):
        R""" Calculate the first order coupling of translations and rotations

        .. math::
            \gamma_1 &= \frac{\langle\Delta r |\Delta\theta| \rangle -
                \langle\Delta r\rangle\langle| \Delta \theta |\rangle }
                {\sqrt{\langle\Delta r^2\rangle\langle\Delta\theta^2\rangle}}

        Args:
            disp_sq (:class:`numpy.array`): Array containing the squared
                displacements
            rotations (class:`numpy.array`): Array of the rotations

        Return:
            float: The coupling of translations and rotations :math:`\gamma_1`
        """
        return ((self.get_mean_trans_rot()
                 - self.get_mean_disp()*self.get_mean_rot())
                / np.sqrt(self.get_msd()*
                          self.get_mean_sq_rot()))


    def get_gamma2(self):
        R""" Calculate the second order coupling of translations and rotations

        .. math:: \gamma_2 &= \frac{\langle(\Delta r \Delta\theta)^2 \rangle -
                \langle\Delta r^2\rangle\langle\Delta \theta^2\rangle
                }{\langle\Delta r^2\rangle\langle\Delta\theta^2\rangle}


        Args:
            disp_sq (:class:`numpy.array`): Array containing the squared
                displacements
            rotations (class:`numpy.array`): Array of the rotations

        Return:
            float: The squared coupling of translations and rotations
            :math:`\gamma_2`
        """
        return ((self.get_mean_sq_trans_rot()
                 - self.get_msd()*self.get_mean_sq_rot())
                / (self.get_msd()
                   * self.get_mean_sq_rot()))

    def get_rot_relax1(self):
        R"""Compute the first rotational relaxation function

        .. math:: C_1(t) = \langle \hat\vec e(0) \cdot
                    \hat \vec e(t) \rangle

        Args:
            rotations (:class:`numpy.array`): Array containing the rotations of
                each molecule

        Return:
            float: The rotational relaxation
        """
        return np.mean(np.cos(self.rotations()))

    def get_rot_relax2(self):
        R"""Compute the second rotational relaxation function

        .. math:: C_1(t) = \langle 2[\hat\vec e(0) \cdot
                    \hat \vec e(t)]^2 - 1 \rangle

        Args:
            rotations (:class:`numpy.array`): Array containing the rotations of
                each molecule

        Return:
            float: The rotational relaxation
        """
        return np.mean(2*np.cos(self.rotations())**2 - 1)

    def get_param_rot(self):
        R"""Compute a parameterised rotational correlation

        """
        return

    def print_all(self, outfile=None):
        """ Print all dynamic quantities to a file

        Prints all the calculated dynamic quantities to either
        stdout or a file. This function only calculates the distances and
        rotations a single time using the private calc methods.

        Args:
            outfile (string): Filename to append to
        """
        output = dict()
        output['msd'] = self.get_msd()
        output['mfd'] = self.get_mfd()
        output['alpha'] = self.get_alpha()
        output['disp'] = self.get_mean_disp()
        output['mean_rot'] = self.get_mean_rot()
        output['time'] = self.timestep()
        output['decoupling'] = self.get_decoupling(0.05, 0.05)
        output['gamma1'] = self.get_gamma1()
        output['gamma2'] = self.get_gamma2()
        output['rot1'] = self.get_rot_relax1()
        output['rot2'] = self.get_rot_relax2()
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
        output['rot1'] = 0
        output['rot2'] = 0
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

