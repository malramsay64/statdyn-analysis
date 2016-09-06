#!/usr/bin/env python
""" A set of classes used for computing the dynamic properties of a Hoomd MD
simulation"""

from __future__ import print_function
import collections
import numpy as np
from TransData import TransData, TransRotData


class CompDynamics(object):
    """ Class to compute the time dependent properties.

    This computes a number of dynamic quantities that are relevant to a
    molecular dynamics simulation.

    Args:
        TData (:class:`TransData.TransData`): A :class:`TransData.TransData`
            object containg the translational motion and time over which
            that translational motion took place.
    """
    def __init__(self, TData):
        assert issubclass(type(TData), TransData), type(TData)
        self.data = TData

    def timestep(self):
        """Compute the timestep difference

        Return:
            int: The number of timesteps the displacement corresponds to
        """
        return self.data.timesteps

    def translations(self):
        """Return the translation of each molecule

        Return:
            :class:`numpy.ndarray`: An array of the translational motion that
            each molecule/particle underwent in a period of time.
        """
        return self.data.trans

    def get_mean_disp(self):
        R""" Compute the mean displacement

        Finds the mean displacement of the molecules using the
        :func:`numpy.mean` function.

        .. math::
            \langle \Delta r \rangle = \langle \sqrt{x^2 + y^2 + z^2} \rangle

        Return:
            float: The mean displacement
        """
        return np.mean(self.translations())

    def get_msd(self):
        R""" Compute the mean squared displacement

        Uses the :func:`numpy.mean` and :func:`numpy.power` functions for
        the computation.

        .. math:: MSD = \langle \Delta r^2 \rangle

        Return:
            float: The mean squared displacement
        """
        return np.mean(np.power(self.translations(), 2))

    def get_mfd(self):
        R""" Compute the mean fourth disaplacement

        Finds the mean of the displacements to the fourth power. Uses the
        :func:`numpy.mean` and :func:`numpy.power` functions for
        the computation.

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
                      {2\langle \Delta r^2  \rangle^2} -1

        Return:
            float: The non-gaussian parameter :math:`\alpha`
        """
        return self.get_mfd()/(2*np.power(self.get_msd(), 2)) - 1

    def get_struct(self, dist=0.3):
        R""" Compute the structural relaxation

        The structural relaxation is given as the proportion of
        particles which have moved further than `dist` from their
        initial positions.

        Args:
            dist (float): The distance cutoff for considering relaxation.
                Defualts to 0.3

        Return:
            float: The structural relaxation of the configuration
        """
        return np.mean(self.translations() < dist)

    def print_all(self, outfile=None):
        R""" Print all dynamic quantities to a file

        Prints all the calculated dynamic quantities to either
        stdout or a file. The output quantities are:

        * timeteps
        * Mean Displacement
        * Mean Squared Displacement (MSD)
        * Mead Fourth Displacement (MFD)
        * Nongaussian parameter (:math:`\alpha`)
        * Structural relaxation

        Args:
            outfile (string): Filename to append output to
        """
        output = collections.OrderedDict()
        output['time'] = self.timestep()
        output['disp'] = self.get_mean_disp()
        output['msd'] = self.get_msd()
        output['mfd'] = self.get_mfd()
        output['alpha'] = self.get_alpha()
        output['struct'] = self.get_struct()
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
        output = collections.OrderedDict()
        output['time'] = 0
        output['disp'] = 0
        output['msd'] = 0
        output['mfd'] = 0
        output['alpha'] = 0
        output['struct'] = 0
        print(keys_to_string(output), file=open(outfile, 'w'))

class CompRotDynamics(CompDynamics):
    """ Class to compute the time dependent characteristics of 2D rigid bodies
    in a hoomd simulation.

    This class extends on from :class:`TimeDep` computing rotational properties
    of the 2D rigid bodies in the system. The distinction of two dimensional
    molecules makes for a simpler analysis of the rotational characteristics.

    Todo:
        Generalise the rotational characteristics to a 3D system.

    Note:
        This class computes positional properties for each rigid body in the
        system. This means that for computations of rigid bodies it would be
        expected to get different translational quantities using the
        :class:`TimeDep` and the :class:`TimeDep2dRigid` classes.

    Args:
        RigidData (:class:`TransData.TransRotData`): A data class containing
            the translational, rotational and time data for all the molecules
            in the system.
    """
    def __init__(self, RigidData=TransRotData()):
        assert (issubclass(type(RigidData), TransRotData)), type(RigidData)
        super(CompRotDynamics, self).__init__(RigidData)
        self.data = RigidData

    def rotations(self):
        R""" Calculate the rotation for every rigid body in the system

        This calculates the angle rotated between the initial configuration and
        the current configuration. It doesn't take into accout rotations past a
        half rotation with values falling in the range :math:`[-\pi,\pi)`.

        Return:
            :class:`numpy.ndarray`: Array of all the rotations
        """
        return self.data.rot


    def get_decoupling(self, delta_disp=0.005, delta_rot=0.005):
        """ Calculates the decoupling of rotations and translations.

        This function performs an intergration over rotational and
        translational space to compute the decoupling of these two
        parameters.

        Note:
            The choice of `delta_disp` and `delta_rot` affect the resulting
            values, especially at small time scales.

        References:
            A. Farone, L. Liu, S.-H. Chen, J. Chem. Phys. 119, 6302 (2003)

        Args:
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


        Return:
            float: The mean rotation
        """
        return np.mean(np.abs(self.rotations()))

    def get_mean_sq_rot(self):
        R""" Compute the mean squared rotation

        Computes the mean of the squared rotation

        .. math:: MSR = \langle \Delta \theta^2 \rangle

        Return:
            float: The mean squared rotation in radians
        """
        return np.mean(np.power(self.rotations(), 2))

    def get_mean_trans_rot(self):
        R""" Compute the coupled translation and rotation

        A measure of the coupling of translations and rotations

        .. math:: \langle \Delta r |\Delta \theta| \rangle

        Return:
            float: The coupling parameter
        """
        return np.mean(self.translations()*np.abs(self.rotations()))

    def get_mean_sq_trans_rot(self):
        R""" Return the squared coupled translation and rotation

        A measure of the coupling of translations and rotations

        .. math::
            \langle \Delta r^2 \Delta \theta^2 \rangle

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

        Return:
            float: The coupling of translations and rotations :math:`\gamma_1`
        """
        return ((self.get_mean_trans_rot()
                 - self.get_mean_disp()*self.get_mean_rot())
                / np.sqrt(self.get_msd()*
                          self.get_mean_sq_rot()))


    def get_gamma2(self):
        R""" Calculate the second order coupling of translations and rotations

        .. math::
            \gamma_2 &= \frac{\langle(\Delta r \Delta\theta)^2 \rangle -
                \langle\Delta r^2\rangle\langle\Delta \theta^2\rangle
                }{\langle\Delta r^2\rangle\langle\Delta\theta^2\rangle}

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

        .. math::
            C_1(t) = \langle \hat\vec e(0) \cdot
                \hat \vec e(t) \rangle

        Return:
            float: The rotational relaxation
        """
        return np.mean(np.cos(self.rotations()))

    def get_rot_relax2(self):
        R"""Compute the second rotational relaxation function

        .. math::
            C_1(t) = \langle 2[\hat\vec e(0) \cdot \
                \hat \vec e(t)]^2 - 1 \rangle

        Return:
            float: The rotational relaxation
        """
        return np.mean(2*np.cos(self.rotations())**2 - 1)

    def get_param_rot(self, alpha=1):
        R"""Compute a parameterised rotational correlation

        .. math::
            \langle \Delta\hat\theta(\alpha)\rangle = \frac{
                \langle|\Delta\theta|e^{\alpha\Delta r} \rangle}{
                \langle e^{\alpha\Delta r} \rangle}

        Args:
            alpha (float): Parameter

        Return:
            float: The computed value

        """
        return (np.mean(np.abs(self.rotations())
                        *np.exp(alpha*self.translations()))
                /np.mean(np.exp(alpha*self.translations())))

    def get_param_trans(self, kappa=1):
        R"""Compute a parameterised translational correlation

        .. math:: \langle \Delta\hat r(\kappa)\rangle = \frac{
            \langle\Delta r e^{\kappa|\Delta \theta|} \rangle}{
            \langle e^{\kappa|\Delta \theta|} \rangle}

        Args:
            kappa (float): Parameter

        Return:
            float: Computed value
        """
        return (np.mean(self.translations()
                        *np.exp(kappa*np.abs(self.rotations())))
                /np.mean(np.exp(kappa*np.abs(self.rotations()))))

    def print_all(self, outfile=None):
        """ Print all dynamic quantities to a file

        Prints all the calculated dynamic quantities to either
        stdout or a file. This function only calculates the distances and
        rotations a single time using the private calc methods.

        Args:
            outfile (string): Filename to append to
        """
        output = collections.OrderedDict()
        output['time'] = self.timestep()
        output['disp'] = self.get_mean_disp()
        output['msd'] = self.get_msd()
        output['mfd'] = self.get_mfd()
        output['alpha'] = self.get_alpha()
        output['mean_rot'] = self.get_mean_rot()
        output['decoupling'] = self.get_decoupling(0.05, 0.05)
        output['gamma1'] = self.get_gamma1()
        output['gamma2'] = self.get_gamma2()
        output['rot1'] = self.get_rot_relax1()
        output['rot2'] = self.get_rot_relax2()
        output['param_rot_n3'] = self.get_param_rot(-3)
        output['param_rot_n2'] = self.get_param_rot(-2)
        output['param_rot_n1'] = self.get_param_rot(-1)
        output['param_rot_n0.1'] = self.get_param_rot(-0.1)
        output['param_rot_0.1'] = self.get_param_rot(0.1)
        output['param_rot_1'] = self.get_param_rot(1)
        output['param_rot_2'] = self.get_param_rot(2)
        output['param_rot_3'] = self.get_param_rot(3)
        output['param_trans_n3'] = self.get_param_trans(-3)
        output['param_trans_n2'] = self.get_param_trans(-2)
        output['param_trans_n1'] = self.get_param_trans(-1)
        output['param_trans_n0.1'] = self.get_param_trans(-0.1)
        output['param_trans_0.1'] = self.get_param_trans(0.1)
        output['param_trans_1'] = self.get_param_trans(1)
        output['param_trans_2'] = self.get_param_trans(2)
        output['param_trans_3'] = self.get_param_trans(3)
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
        output = collections.OrderedDict()
        output['time'] = 0
        output['disp'] = 0
        output['msd'] = 0
        output['mfd'] = 0
        output['alpha'] = 0
        output['mean_rot'] = 0
        output['decoupling'] = 0
        output['gamma1'] = 0
        output['gamma2'] = 0
        output['rot1'] = 0
        output['rot2'] = 0
        output['param_rot_n3'] = 0
        output['param_rot_n2'] = 0
        output['param_rot_n1'] = 0
        output['param_rot_n0.1'] = 0
        output['param_rot_0.1'] = 0
        output['param_rot_1'] = 0
        output['param_rot_2'] = 0
        output['param_rot_3'] = 0
        output['param_trans_n3'] = 0
        output['param_trans_n2'] = 0
        output['param_trans_n1'] = 0
        output['param_trans_n0.1'] = 0
        output['param_trans_0.1'] = 0
        output['param_trans_1'] = 0
        output['param_trans_2'] = 0
        output['param_trans_3'] = 0
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

