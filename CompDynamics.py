#!/usr/bin/env python3
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
        self.data = TData
        self._d_disp_compute = 0
        self._d_disp2_compute = 0
        self._d_disp4_compute = 0
        self._output_all = collections.OrderedDict()
        self._output_all['time'] = self.timestep
        self._output_all['disp'] = self.get_mean_disp
        self._output_all['msd'] = self.get_msd
        self._output_all['mfd'] = self.get_mfd
        self._output_all['alpha'] = self.get_alpha
        self._output_all['struct'] = self.get_struct

    def timestep(self):
        """Compute the timestep difference

        Return:
            int: The number of timesteps the displacement corresponds to
        """
        return self.data.time_diff

    def translations(self):
        """Return the translation of each molecule

        Return:
            :class:`numpy.ndarray`: An array of the translational motion that
            each molecule/particle underwent in a period of time.
        """
        return self.data['displacement']

    def _d_disp(self):
        """ Internal funtion to compute mean displacement
        """
        if self._d_disp_compute == 0:
            self._d_disp_compute = np.mean(self.translations())
        return self._d_disp_compute

    def _d_disp2(self):
        """Internal function to compute mean squared disp
        """
        if self._d_disp2_compute == 0:
            self._d_disp2_compute = np.mean(np.power(self.translations(), 2))
        return self._d_disp2_compute

    def _d_disp4(self):
        """Internal function to compute mean fourth disp
        """
        if self._d_disp4_compute == 0:
            self._d_disp4_compute = np.mean(np.power(self.translations(), 4))
        return self._d_disp4_compute

    def get_mean_disp(self):
        R""" Compute the mean displacement

        Finds the mean displacement of the molecules using the
        :func:`numpy.mean` function.

        .. math::
            \langle \Delta r \rangle = \langle \sqrt{x^2 + y^2 + z^2} \rangle

        Return:
            float: The mean displacement
        """
        return self._d_disp()

    def get_msd(self):
        R""" Compute the mean squared displacement

        Uses the :func:`numpy.mean` and :func:`numpy.power` functions for
        the computation.

        .. math:: MSD = \langle \Delta r^2 \rangle

        Return:
            float: The mean squared displacement
        """
        return self._d_disp2()

    def get_mfd(self):
        R""" Compute the mean fourth disaplacement

        Finds the mean of the displacements to the fourth power. Uses the
        :func:`numpy.mean` and :func:`numpy.power` functions for
        the computation.

        .. math:: MFD = \langle \Delta r^4 \rangle

        Return:
            float: The mean fourth displacement
        """
        return self._d_disp4()

    def get_alpha(self):
        R""" Compute the non-gaussian parameter :math:`\alpha`

        The non-gaussian parameter is given as

        .. math::
            \alpha = \frac{\langle \Delta r^4\rangle}
                      {2\langle \Delta r^2  \rangle^2} -1

        Return:
            float: The non-gaussian parameter :math:`\alpha`
        """
        return self._d_disp4()/(2*np.power(self._d_disp2(), 2)) - 1

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
        string = ' '.join(
            [func() for func in self._output_all.values()])
        if outfile:
            print(string, file=open(outfile, 'a'))
        else:
            print(string)

    def print_heading(self, outfile):
        """ Write heading values to outfile which match up with the values given
        by print_all().

        Args:
            outfile (string): Filename to write headings to
        """
        string = ' '.join(
            [key for key in self._output_all.keys()])
        print(string, file=open(outfile, 'w'))


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

    def __init__(self, RigidData):
        super(CompRotDynamics, self).__init__(RigidData)
        self.data = RigidData
        self._d_theta_compute = 0
        self._d_theta2_compute = 0
        self._d_disp_d_theta_compute = 0
        self._d_disp2_d_theta2_compute = 0
        self._output_all = collections.OrderedDict()
        self._output_all['time'] = self.timestep
        self._output_all['disp'] = self.get_mean_disp
        self._output_all['msd'] = self.get_msd
        self._output_all['mfd'] = self.get_mfd
        self._output_all['alpha'] = self.get_alpha
        self._output_all['mean_rot'] = self.get_mean_rot
        self._output_all['mean_rot2'] = self._d_theta2
        self._output_all['mean_trans_rot'] = self._d_disp_d_theta
        self._output_all['mean_trans2_rot2'] = self._d_disp2_d_theta2
        self._output_all['gamma1'] = self.get_gamma1
        self._output_all['gamma2'] = self.get_gamma2
        self._output_all['gamma_mobile'] = self.get_gamma_fast
        self._output_all['rot1'] = self.get_rot_relax1
        self._output_all['rot2'] = self.get_rot_relax2
        self._output_all['struct'] = self.get_struct
        self._output_all['COM_struct'] = self.get_com_struct
        self._output_all['trans_corel'] = self.get_trans_correl
        self._output_all['rot_corel'] = self.get_rot_correl

    def rotations(self):
        R""" Calculate the rotation for every rigid body in the system

        This calculates the angle rotated between the initial configuration and
        the current configuration. It doesn't take into accout rotations past a
        half rotation with values falling in the range :math:`[-\pi,\pi)`.

        Return:
            :class:`numpy.ndarray`: Array of all the rotations
        """
        return self.data['rotation'][:self.data.bodies]

    def translations(self):
        """Return the translation of each molecule

        Return:
            :class:`numpy.ndarray`: An array of the translational motion that
            each molecule/particle underwent in a period of time.
        """
        return self.data['displacement'][:self.data.bodies]

    def _all_translations(self):
        """Return the translation of every particle

        Return:
            :class:`numpy.ndarray`: An array of the translational motion that
            each particle underwent in a period of time.
        """
        return self.data['displacement']

    def _d_theta(self):
        """Compute the mean rotation"""
        if self._d_theta_compute == 0:
            self._d_theta_compute = np.mean(np.abs(self.rotations()))
        return self._d_theta_compute

    def _d_theta2(self):
        """Compute the mean squared rotation"""
        if self._d_theta2_compute == 0:
            self._d_theta2_compute = np.mean(np.power(self.rotations(), 2))
        return self._d_theta2_compute

    def _d_disp_d_theta(self):
        """Compute dr dtheta"""
        if self._d_disp_d_theta_compute == 0:
            self._d_disp_d_theta_compute = np.mean(self.translations()
                                                   * np.abs(self.rotations()))
        return self._d_disp_d_theta_compute

    def _d_disp2_d_theta2(self):
        "Compute dr2 dtheta2"""
        if self._d_disp2_d_theta2_compute == 0:
            self._d_disp2_d_theta2_compute = np.mean(
                np.power(self.translations(), 2)
                * np.power(self.rotations(), 2)
            )
        return self._d_disp2_d_theta2_compute

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
            delta_disp (float): The bin size of the displacement for
                integration
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

        prob = normalise_probability(
            np.asmatrix(prob),
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
        return self._d_theta()

    def get_mean_sq_rot(self):
        R""" Compute the mean squared rotation

        Computes the mean of the squared rotation

        .. math:: MSR = \langle \Delta \theta^2 \rangle

        Return:
            float: The mean squared rotation in radians
        """
        return self._d_theta2()

    def get_mean_trans_rot(self):
        R""" Compute the coupled translation and rotation

        A measure of the coupling of translations and rotations

        .. math:: \langle \Delta r |\Delta \theta| \rangle

        Return:
            float: The coupling parameter
        """
        return self._d_disp_d_theta()

    def get_mean_sq_trans_rot(self):
        R""" Return the squared coupled translation and rotation

        A measure of the coupling of translations and rotations

        .. math::
            \langle \Delta r^2 \Delta \theta^2 \rangle

        Return:
            float: The squared coupling of translations and rotations
        """
        return self._d_disp2_d_theta2()

    def get_gamma1(self):
        R""" Calculate the first order coupling of translations and rotations

        .. math::
            \gamma_1 &= \frac{\langle\Delta r |\Delta\theta| \rangle -
                \langle\Delta r\rangle\langle| \Delta \theta |\rangle }
                {\sqrt{\langle\Delta r^2\rangle\langle\Delta\theta^2\rangle}}

        Return:
            float: The coupling of translations and rotations :math:`\gamma_1`
        """
        return ((self._d_disp_d_theta() - self._d_disp() * self._d_theta()) /
                (self._d_disp() * self._d_theta()))

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
        return ((self._d_disp2_d_theta2()
                 - self._d_disp2() * self._d_theta2()) /
                (self._d_disp2() * self._d_theta2()))

    def get_gamma_fast(self, fraction=0.1):
        R""" Calculate the second order coupling of highly mobile molecules

        .. math::
            \gamma_2 &= \frac{\langle(\Delta r \Delta\theta)^2 \rangle -
                \langle\Delta r^2\rangle\langle\Delta \theta^2\rangle
                }{\langle\Delta r^2\rangle\langle\Delta\theta^2\rangle}

        Args:
            fraction (float): the fraction of molecules that are considered
                highly mobile. Defaults to 0.1

        Return:
            float: The squared coupling of translations and rotations of
                the top `fraction` of particles

        This uses the fraction of both highly rotationally mobile and
        highly translationally mobile molecules so the complete set of
        molecules studied is likely over the fraction given.
        """
        num_mobile = int(len(self._d_disp2())*fraction)
        mobile_disp = np.argpartition(self._d_disp2(), num_mobile)[-num_mobile:]
        mobile_rot = np.argpartition(self._d_theta2(), num_mobile)[-num_mobile:]
        mobile = np.union1d(mobile_disp, mobile_rot)
        translations2 = np.power(self.translations()[mobile], 2)
        rotations2 = np.power(self.rotations()[mobile], 2)
        return ((np.mean(translations2*rotations2)
                 - np.mean(translations2)*np.mean(rotations2))
                / (np.mean(translations2) * np.mean(rotations2)))

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
                        * np.exp(alpha * self.translations()))
                / np.mean(np.exp(alpha * self.translations())))

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
                        * np.exp(kappa * np.abs(self.rotations())))
                / np.mean(np.exp(kappa * np.abs(self.rotations()))))

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
        return np.mean(self._all_translations() < dist)

    def get_com_struct(self, dist=0.3):
        R""" Compute the structural relaxation for the Centers of mass

        The structural relaxation of the Centers of Mass (COM) is given as the
        proportion of molecules that have moved further than `dist` from their
        initial positions.

        Args:
            dist (float): The distance cutoff for considering relaxation.
                Defualts to 0.3

        Return:
            float: The structural COM relaxation of the configuration
        """
        return np.mean(self.translations() < dist)

    def get_trans_correl(self):
        r""" Compute the correlation of rotations and translations

        This correlation factor is given by the equation

        .. math:
            \frac{\langle \Delta r \Delta \theta \rangle
                - \langle \Delta r \rangle \langle \Delta \theta \rangle
                }{
                \frac {
                    \langle \Delta r \rangle \langle \Delta \theta^2 \rangle
                    }{
                    \langle \Delta \theta \rangle
                    }
                - \langle \Delta r \rangle \langle \Delta\theta\rangle
                }

        Return:
            float: the rotational correlation of the configuration
        """
        return ((self._d_disp2() - np.power(self._d_disp(), 2))
                / np.power(self._d_disp(), 2))

    def get_rot_correl(self):
        r""" Compute the correlation of rotations and translations

        This correlation factor is given by the equation

        .. math:
            \frac{ \langle \Delta r \Delta \theta \rangle
                - \langle \Delta r \rangle \langle \Delta \theta \rangle
                }{
                \frac {
                    \langle \Delta \theta \rangle \langle \Delta r^2 \rangle
                    }{
                    \langle \Delta r \rangle
                    }
                - \langle \Delta r \rangle \langle \Delta\theta\rangle
                }

        Return:
            float: the rotational correlation of the configuration
        """
        return ((self._d_theta2() - np.power(self._d_theta(), 2))
                / np.power(self._d_theta(), 2))

    def print_all(self, outfile=None):
        """ Print all dynamic quantities to a file

        Prints all the calculated dynamic quantities to either
        stdout or a file. This function only calculates the distances and
        rotations a single time using the private calc methods.

        Args:
            outfile (string): Filename to append to
        """
        string = ' '.join(
            [str(func()) for func in self._output_all.values()])
        if outfile:
            print(string, file=open(outfile, 'a'))
        else:
            print(string)

    def print_heading(self, outfile):
        """ Write heading values to outfile which match up with the values given
        by print_all().

        Args:
            outfile (string): Filename to write headings to
        """
        string = ' '.join(
            [key for key in self._output_all.keys()])
        print((string), file=open(outfile, 'w'))


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
