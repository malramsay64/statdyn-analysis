#!/usr/bin/env python
""" A set of classes used for computing the dynamic properties of a Hoomd MD
simulation"""

import os
import math
import numpy as np
from hoomd_script import init, update, pair, integrate, analyze, group, run_upto
import stepSize

class TimeDep(object):
    """ Class to compute the time dependent characteristics of individual
    particles in a hoomd simulation."""
    rigid = False
    def __init__(self, system):
        self.t_init = system.take_snapshot(rigid_bodies=self.rigid)
        self.pos_init = unwrap(self.t_init, self.rigid)
        self.timestep = system.get_metadata()['timestep']

    def get_time_diff(self, timestep):
        """ Returns the difference in time between the currrent timestep and the
        initial timestep.
        param: timestep The timestep the difference is to be calculated at
        """
        return timestep - self.timestep

    def get_displacement_sq(self, snapshot):
        """ Calculate the squared displacement for all bodies in the system.
        This is the single function that returns displacements for all the
        other computations. The squared displacement is found as the sqrt
        operation is relatively slow and not all functions require it, e.g. msd.
        param: snapshot The configuration at which the difference is computed

        return: Array of the squared displacements
        """
        curr = unwrap(snapshot, self.rigid)
        return np.power(curr - self.pos_init, 2).sum(axis=1)

    def _calc_mean_disp(self, displacement_sq):
        """ Calculates the mean displacement for all bodies in the
        system.
        param: displacement_sq The squared displacement of all the particles
        used to calculate the mean displacement
        """
        return np.mean(np.sqrt(displacement_sq))

    def get_mean_disp(self, snapshot):
        """ Calculates the mean displacement for all rigid bodies in the
        system.
        param: snapshot The configuration used to calculate the displacement
        """
        return self._calc_mean_disp(self.get_displacement_sq(snapshot))

    def _calc_msd(self, displacement_sq):
        """ Calculate the mean squared displacement of particles
        param: displacement_sq The squared displacements of the particles
        """
        return np.mean(displacement_sq)

    def get_msd(self, snapshot):
        """ Return the mean squared displacement of particles
        param: snapshot The snapshot the particles have moved to
        """
        return self._calc_msd(self.get_displacement_sq(snapshot))

    def _calc_mfd(self, displacement_sq):
        """ Calculate the mean squared displacement of particles
        param: displacement_sq The squared displacements of the particles
        """
        return np.mean(np.power(displacement_sq, 2))

    def get_mfd(self, snapshot):
        """ Return the mean fouth (quartic) displacement of particles
        param: snapshot The snapshot the particles have moved to
        """
        return self._calc_mfd(self.get_displacement_sq(snapshot))

    def _calc_alpha(self, displacement_sq):
        """ Calculate the non-gaussian parameter
        param: snapshot The configuration the particles have moved to
        """
        msd = self._calc_msd(displacement_sq)
        mfd = self._calc_mfd(displacement_sq)
        return mfd/(2.*(msd*msd))

    def get_alpha(self, snapshot):
        """ Return the non-gaussian parameter
        param: snapshot The configuration the particles have moved to
        """
        return self._calc_alpha(self.get_displacement_sq(snapshot))


class TimeDep2dRigid(TimeDep):
    """ Class to compute the time dependent characteristics of rigid bodies
    in a hoomd simulation.
    param: system The initial system configuration
    """
    rigid = True
    def __init__(self, system):
        super().__init__(system)


    def get_rot(self, snapshot):
        """ Calculate the mean rotation of rigid bodies in the system
        param: snapshot configuration from which to calulcate the motion
        """
        rot = self.get_rotations(snapshot)
        return np.mean(rot, dtype=np.float64)

    def get_rotations(self, snapshot):
        """ Calculate the rotation for every rigid body in the system. This
        doesn't take into accout multiple rotations with values falling between
        -pi and pi.
        param: snapshot The configuration from which to calculate the rotational
        change.
        """
        rot = np.empty(len(self.t_init.bodies.com))
        for i in range(len(self.t_init.bodies.com)):
            rot[i] = quat_to_2d(self.t_init.bodies.orientation[i]) -\
                     quat_to_2d(snapshot.bodies.orientation[i])
            if rot[i] > math.pi:
                rot[i] = 2*math.pi - rot[i]
        return rot


    def _calc_decoupling(self, snapshot, delta_disp, delta_rot):
        """ Calculates the coupling strength parameter of the translational and
        rotational motion as described by [[|Farone and Chen]].
        :param snapshot The snapshot with which to take the distances and
        rotations from the initial.
        :param dr The size on the binning for the translational motion.
        :param dtheta The size of the binning for the rotational motion.
        """
        # Calculate and bin displacements
        disp = np.sqrt(self.get_displacement_sq(snapshot))
        disp = np.floor(disp/delta_disp).astype(int)
        # adding 1 to account for 0 value
        disp_max = np.max(disp+1)
        disp_array = np.asmatrix(np.power(\
                np.arange(1, disp_max+1)*delta_disp, 2))
        # Calculate and bin rotaitons
        rot = self.get_rotations(snapshot)
        rot = np.floor(np.abs(rot)/delta_rot).astype(int)
        # adding 1 to account for 0 value
        rot_max = np.max(rot+1)
        rot_array = np.asmatrix(np.sin(\
                np.arange(1, rot_max+1)*delta_rot))
        # Use binned values to create a probability matrix
        prob = np.zeros((rot_max, disp_max))
        for i, j in zip(rot, disp):
            prob[i][j] += 1

        prob = np.asmatrix(prob)
        # Calculate tranlational and rotational probabilities
        p_trans = (prob.transpose() * rot_array.transpose())
        p_trans *= delta_rot
        p_rot = (prob * disp_array.transpose())
        p_rot *= delta_disp

        # Calculate the squared difference between the combined and individual
        # probabilities and then integrate over the differences to find the
        # coupling strength
        diff2 = np.power(prob - p_rot * p_trans.transpose(), 2)
        decoupling = (diff2 * np.power(disp_array, 2).transpose()) \
                * np.power(rot_array, 2).sum()
        decoupling /= ((prob*disp_array.transpose()) * rot_array).sum()
        return decoupling.sum()

    def get_decoupling(self, snapshot, delta_disp=0.005, delta_rot=0.005):
        """ Returns the coupling strength parameter of the translational and
        rotational motion as described by [[|Farone and Chen]].
        :param snapshot The snapshot with which to take the distances and
        rotations from the initial.
        :param dr The size on the binning for the translational motion. This is
        set to 0.005 by default.
        :param dtheta The size of the binning for the rotational motion. This
        is set to 0.005 by default.
        """
        return self._calc_decoupling(snapshot, \
                                      delta_disp,\
                                      delta_rot \
                                     )

    def print_all(self, snapshot, timestep, outfile=None):
        """ Function to print all the calculated dynamic quantities to either
        stdout or a function. This function only calculates the distances and
        rotations a single time.
        param: snapshot The configuration the bodies have moved to. This
        snapshot requires that the rigid_bodies=True is passed to the
        take_snapshot function.
        param: timestep The current timestep of the simulaton. The same timestep
        as the snapshot.
        param: outfile Filename of file to write output to. If no file specified
        the data is ouput to stdout.
        """
        disp_sq = self.get_displacement_sq(snapshot)
        msd = self._calc_msd(disp_sq)
        mfd = self._calc_mfd(disp_sq)
        alpha = self._calc_alpha(disp_sq)
        disp = self._calc_mean_disp(disp_sq)
        rot = self.get_rot(snapshot)
        time = self.get_time_diff(timestep)
        decoupling = self.get_decoupling(snapshot)
        if outfile:
            print(time, rot, disp, msd, mfd, alpha, decoupling, \
                    file=open(outfile, 'a'))
        else:
            print(time, rot, disp, msd, mfd, alpha, decoupling)

    def print_heading(self, outfile):
        """ Write heading values to outfile which match up with the values given
        by print_all().
        """
        print("time", "rotation", "displacement", "msd", "mfd",\
                "alpha", "coupling", file=open(outfile, 'w'))

def quat_to_2d(quat):
    """ Convert the quaternion representation of angle to a two dimensional
    angle in the xy plane.
    param: quat Quaternion for conversion
    """
    return math.atan2(quat.x*quat.w + quat.y*quat.z,\
            0.5-quat.y*quat.y - quat.z-quat.z)

def scalar3_to_array(scalar):
    """ Convert cuda scalar3 representation to a numpy array
    param: scalar Scalar3 for conversion
    """
    return np.array([scalar.x, scalar.y, scalar.z])

def scalar4_to_array(scalar):
    """ Convert cuda scalar4 to a numpy array
    param: scalar Scalar4 for conversion
    """
    return np.array([scalar.x, scalar.y, scalar.z, scalar.w])

def unwrap(snapshot, rigid):
    """ Function to unwrap the periodic distances in the snapshots to
    discreete distances that are easy to use for computing distance.
    param: snapshot Snapshot containing the data to unwrap
    param: rigid Boolean value indicating whether we are unwrapping rigid
    body centers of mass or particle positions.
    """
    box_dim = np.array([snapshot.box.Lx,\
                        snapshot.box.Ly,\
                        snapshot.box.Lz \
                       ])
    if rigid:
        pos = np.array([scalar3_to_array(i) for i in snapshot.bodies.com])
        image = np.array([scalar3_to_array(i) \
                for i in snapshot.bodies.body_image])
    else:
        pos = np.array(snapshot.particles.position)
        image = np.array(snapshot.particles.image)
    return pos + image*box_dim


def compute_dynamics(input_xml,
                     temp,
                     press,
                     steps,
                     rigid=True):
    """ Run a hoomd simulation calculating the dynamic quantites on a power
    law scale such that both short timescale and long timescale events are
    vieable on the same figure while retaining a reasonable runtime.
    param: input_xml Filename of the file containing the input configuration
    for the simulation
    param: temp The target temperature at which to run the simulation
    param: press The target pressure at which to run the simulation
    param: rigid Boolean value indicating whether to integrate using rigid
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

    thermo = analyze.log(filename=basename+"-thermo.dat", \
                         quantities=['temperature', 'pressure', \
                                     'potential_energy', \
                                     'rotational_kinetic_energy', \
                                     'translational_kinetic_energy'
                                    ], \
                         period=1000)

    # Initialise dynamics quantities
    if rigid:
        dyn = TimeDep2dRigid(system)
        dyn.print_heading(basename+"-dyn.dat")

    timestep = 0
    step_iter = stepSize.PowerSteps()


    while timestep < steps:
        timestep = step_iter.next()
        run_upto(timestep)
        dyn.print_all(system.take_snapshot(rigid_bodies=True), \
                      timestep, \
                      outfile=basename+"-dyn.dat" \
                     )
        thermo.query('pressure')


