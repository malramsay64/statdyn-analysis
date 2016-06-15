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
    def __init__(self, system):
        self.t_init = system.take_snapshot()
        self.timestep = system.get_metadata()['timestep']
        self.box_dim = np.array([self.t_init.box.Lx,\
                                 self.t_init.box.Ly,\
                                 self.t_init.box.Lz
                                ])

    def get_time_diff(self, timestep):
        """ Returns the difference in time between the currrent timestep and the
        initial timestep.
        param: timestep The timestep the difference is to be calculated at
        """
        return timestep - self.timestep

    def get_msd(self, snapshot):
        """ Calculate the mean squared displacement of particles
        param: snapshot The snapshot the particles have moved to
        """
        snapshot = snapshot
        msd = np.empty(len(self.t_init.particles.position))
        sys_box_dim = np.array([snapshot.box.Lx,\
                                snapshot.box.Ly,\
                                snapshot.box.Lz\
                               ])
        for i in range(len(self.t_init.particles.position)):
            msd[i] = sum(np.power(\
                ((self.t_init.particles.image[i]*self.box_dim \
                + self.t_init.particles.position[i]) \
                - (snapshot.particles.image[i]*sys_box_dim \
                + snapshot.particles.position[i]))\
                .astype(np.float65), 2))
        return np.mean(msd, dtype=np.float64)

    def get_mfd(self, snapshot):
        """ Calculate the mean fouth (quartic) displacement of particles
        param: snapshot The snapshot the particles have moved to
        """
        snapshot = snapshot
        mfd = np.empty(len(self.t_init.particles.position))
        sys_box_dim = np.array([snapshot.box.Lx,\
                                snapshot.box.Ly,\
                                snapshot.box.Lz
                               ])
        for i in range(len(self.t_init.particles.position)):
            mfd[i] = sum(np.power(\
                ((self.t_init.particles.image[i]*self.box_dim \
                + self.t_init.particles.position[i]) \
                - (snapshot.particles.image[i]*sys_box_dim\
                + snapshot.particles.position[i])) \
                .astype(np.float64), 4))
        return np.mean(mfd, dtype=np.float64)

    def get_alpha(self, snapshot):
        """ Calculate the non-gaussian parameter
        param: snapshot The configuration the particles have moved to
        """
        msd = self.get_msd(snapshot)
        mfd = self.get_mfd(snapshot)
        return mfd/(2.*(msd*msd))


class TimeDep2dRigid(object):
    """ Class to compute the time dependent characteristics of rigid bodies
    in a hoomd simulation.
    param: system The initial system configuration
    """
    def __init__(self, system):
        self.t_init = system.take_snapshot(rigid_bodies=True)
        self.timestep = system.get_metadata()['timestep']
        self.box_dim = np.array([self.t_init.box.Lx, \
                                 self.t_init.box.Ly, \
                                 self.t_init.box.Lz  \
                                ])

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
        disp = np.empty(len(self.t_init.bodies.com))
        sys_box_dim = np.array([snapshot.box.Lx, \
                                snapshot.box.Ly, \
                                snapshot.box.Lz  \
                               ])
        for i in range(len(self.t_init.bodies.com)):
            disp[i] = sum(np.power(\
              ((scalar3_to_array(self.t_init.bodies.com[i]) \
              + scalar3_to_array(self.t_init.bodies.body_image[i])*self.box_dim)\
              - scalar3_to_array(snapshot.bodies.com[i]) \
              + scalar3_to_array(snapshot.bodies.body_image[i]) * sys_box_dim)\
              .astype(np.float64), 2))
            return disp

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

    def get_msd(self, snapshot):
        """ Calculates the mean squared displacement for all rigid bodies in the
        system.
        param: snapshot The configuration used to calculate the displacement
        """
        return np.mean(self.get_displacement_sq(snapshot))

    def get_mfd(self, snapshot):
        """ Calculates the mean fourth (quartic) displacement for all rigid
        bodies in the system.
        param: snapshot The configuration used to calculate the displacement
        """
        return np.mean(np.power(self.get_displacement_sq(snapshot), 2))

    def get_mean_disp(self, snapshot):
        """ Calculates the mean displacement for all rigid bodies in the
        system.
        param: snapshot The configuration used to calculate the displacement
        """
        return np.mean(np.sqrt(self.get_displacement_sq(snapshot)))

    def get_alpha(self, snapshot):
        """ Calculate the non-gaussian parameter
        param: snapshot The configuration the particles have moved to
        """
        disp_sq = self.get_displacement_sq(snapshot)
        msd = np.mean(disp_sq)
        mfd = np.mean(np.power(disp_sq), 2)
        return mfd/(2*msd*msd) - 1

    def get_decoupling(self, snapshot, delta_disp=0.005, delta_rot=0.005):
        """ Calculates the coupling strength parameter of the translational and
        rotational motion as described by [[|Farone and Chen]].
        :param snapshot The snapshot with which to take the distances and
        rotations from the initial.
        :param dr The size on the binning for the translational motion. This is
        set to 0.005 by default.
        :param dtheta The size of the binning for the rotational motion. This
        is set to 0.005 by default.
        """
        # Calculate and bin displacements
        disp = np.sqrt(self.get_displacement_sq(snapshot))
        disp = np.floor(disp/delta_disp)
        disp_max = np.max(disp)
        disp_array = np.power(np.arrange(delta_disp, delta_disp*disp_max, delta_disp), 2)
        # Calculate and bin rotaitons
        rot = self.get_rotations(snapshot)
        rot = np.floor(np.abs(rot)/delta_rot)
        rot_max = np.max(rot)
        rot_array = np.sin(np.arrange(delta_rot, delta_rot*rot_max, delta_rot))
        # Use binned values to create a probability matrix
        prob = np.zeros((rot_max, disp_max))
        for i, j in zip(rot, prob):
            prob[i][j] += 1

        # Calculate tranlational and rotational probabilities
        p_trans = (prob * rot_array).sum(axis=1)*delta_rot
        p_rot = (prob.transpose()*np.power(disp_array, 2)).sum(axis=1)*delta_disp

        # Calculate the squared difference between the combined and individual
        # probabilities and then integrate over the differences to find the
        # coupling strength
        diff2 = np.power(prob - p_trans.transpose * p_rot, 2)
        coupling = ((diff2*np.power(rot_array, 2)).transpose() \
                * np.power(disp_array, 2))
        coupling /= ((prob*rot_array).transpose() * disp_array)

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
        msd = np.mean(disp_sq)
        mfd = np.mean(np.power(disp_sq, 2))
        alpha = mfd/(2*msd**2) - 1
        rot = self.get_rot(snapshot)
        disp = np.mean(np.sqrt(disp_sq))
        time = self.get_time_diff(timestep)
        if outfile:
            write_file = open(outfile, 'a')
            write_file.write(\
                    str(time)+" "+str(rot)+" "+str(disp)+" "+str(msd)+" "\
                    +str(mfd)+" "+str(alpha)+"\n")
        else:
            print(time, rot, disp, msd, mfd, alpha)

    def print_heading(self, outfile):
        """ Write heading values to outfile which match up with the values given
        by print_all().
        """
        write_file = open(outfile, 'w')
        write_file.write("time rotation displacement msd mfd alpha\n")


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


def compute_dynamics(input_xml,
                     temp,
                     press,
                     steps,
                     potentials=None,
                     rigid=True):
    """ Run a hoomd simulation calculating the dynamic quantites on a power
    law scale such that both short timescale and long timescale events are
    vieable on the same figure while retaining a reasonable runtime.
    param: input_xml Filename of the file containing the input configuration
    for the simulation
    param: temp The target temperature at which to run the simulation
    param: press The target pressure at which to run the simulation
    param: potentials An alternate interaction potential for the simulation.
    If none the default interaction potential is used.
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

    if not potentials:
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
    step_iter = stepSize.powerSteps()


    while timestep < steps:
        timestep = step_iter.next()
        run_upto(timestep)
        dyn.print_all(system.take_snapshot(rigid_bodies=True), \
                      timestep,
                      basename+"-dyn.dat"
                     )


