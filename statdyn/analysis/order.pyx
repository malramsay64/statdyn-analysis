#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Perform analysis on the trajectories from the simulation."""

from typing import List, Set

import cython
import numpy as np
from freud.box import Box
from freud.locality import NearestNeighbors
from quaternion import as_quat_array, as_rotation_vector

from statdyn.molecules import Molecule, Trimer

from ..molecules import Molecule, Trimer

cimport numpy as np
from libc cimport math

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_neighbours(np.ndarray[float, ndim=1] box,
                         np.ndarray[float, ndim=2] position,
                         float max_radius,
                         unsigned int max_neighbours):
    """Compute the neighbour list."""
    ndim = 2
    if ndim == 2:
        simulation_box = Box(box[0], box[1], is2D=True)
    else:
        simulation_box = Box(box[0], box[1], box[2], is2D=False)

    cdef np.ndarray[unsigned int, ndim=2] neighs
    neighs = np.empty((position.shape[0], max_neighbours), dtype=np.uint32)

    nn = NearestNeighbors(rmax=max_radius, n_neigh=max_neighbours, strict_cut=True)
    nn.compute(simulation_box, position, position)

    neighs[...] = nn.getNeighborList()[...]
    return neighs

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_mol_neighbours(np.ndarray[unsigned int, ndim=2] neighbourlist,
                             np.ndarray[unsigned int, ndim=1] bodies,
                             unsigned int max_neighbours=7):
    """Convert a particle neighbourlist to a molecular neighbourlist."""
    cdef unsigned int nmols = bodies.max() + 1
    cdef unsigned int mol_index
    cdef unsigned int no_value = np.iinfo(np.uint32).max
    cdef unsigned int num_particles = bodies.shape[0]
    cdef unsigned int num_neighs = neighbourlist.shape[1]
    cdef unsigned int i, n, atom_index
    cdef np.ndarray[unsigned int, ndim=2] mol_neighbours

    mol_neighbours = np.full((nmols, max_neighbours), no_value, dtype=np.uint32)

    with nogil:
        for atom_index in range(num_particles):
            mol_index = bodies[atom_index]
            for n in range(num_neighs):
                for i in range(max_neighbours):
                    # We reach the end of the added neighbours without finding ourself
                    if mol_neighbours[mol_index, i] == no_value:
                        mol_neighbours[mol_index, i] = neighbourlist[atom_index, n]
                        break
                        # We already exist in the list
                    elif mol_neighbours[mol_index, i] == neighbourlist[atom_index, n]:
                        break
    return mol_neighbours

@cython.boundscheck(False)
@cython.wraparound(False)
def num_neighbours(np.ndarray[float, ndim=1] box,
                   np.ndarray[float, ndim=2] position,
                   np.ndarray[float, ndim=2] orientation,
                   float max_radius=3.5):
    cdef unsigned int max_neighbours = 8
    cdef unsigned int num_mols = position.shape[0]
    cdef unsigned int no_value = np.iinfo(np.uint32).max
    cdef np.ndarray[unsigned int, ndim=2] neighbourlist
    cdef np.ndarray[unsigned int, ndim=1] n_neighs

    n_neighs = np.zeros(num_mols, dtype=np.uint32)
    neighbourlist = compute_neighbours(box, position, max_radius, max_neighbours)

    for i in range(num_mols):
        for k in range(max_neighbours):
            if neighbourlist[i, k] < num_mols:
                n_neighs[i] += 1
            else:
                break
    return n_neighs


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef orientational_order(np.ndarray[float, ndim=1] box: np.ndarray,
                        np.ndarray[float, ndim=2] position: np.ndarray,
                        np.ndarray[float, ndim=2] orientation: np.ndarray,
                        float max_radius=3.5):

    cdef unsigned int no_value = np.iinfo(np.uint32).max
    cdef unsigned int max_neighbours = 8
    cdef unsigned int num_mols = position.shape[0]
    cdef unsigned int mol_index, n, num_neighbours, curr_neighbour

    cdef np.ndarray[float, ndim=1] angles
    cdef np.ndarray[unsigned int, ndim=2] neighbourlist
    cdef np.ndarray[float, ndim=1] order_parameter

    angles = get_z_orientation(orientation)
    neighbourlist = compute_neighbours(box, position, max_radius, max_neighbours)
    order_parameter = np.zeros(num_mols, dtype=np.float32)

    with nogil:
        for mol_index in range(num_mols):
            num_neighbours = 0
            for n in range(max_neighbours):
                curr_neighbour = neighbourlist[mol_index, n]
                if curr_neighbour == no_value:
                    break
                order_parameter[mol_index] += math.fabs(math.cos(angles[mol_index] - angles[curr_neighbour]))
                num_neighbours += 1
            if num_neighbours > 1:
                order_parameter[mol_index] /= num_neighbours
            else:
                order_parameter[mol_index] = 0
    return order_parameter



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[float, ndim=1] get_z_orientation(np.ndarray[float, ndim=2] orientations):
    r"""Get the z component of the molecular orientation.

    This function converts the quaternion representing the orientation to a
    rotation vector from which the z component is extracted. The function bounds
    the rotation to the range $[-\pi, \pi)$.

    Args:
        orientations (py:class:`numpy.ndarray`): The orientations as an array of
            the quaternion values in an n x 4 array with the quaternion
            represente in the form (w, x, y, z).

    Returns:
        py:class:`numpy.ndarray`: A vector with the same length as the input
            vector which contains the z component of the orientation.

    """
    cdef np.ndarray[float, ndim=1] angles
    cdef float pi = np.pi, tau = 2*np.pi
    cdef unsigned int i, num_elements
    num_elements = orientations.shape[0]

    angles = as_rotation_vector(as_quat_array(orientations))[:, 2].astype(np.float32)

    for i in range(num_elements):
        if angles[i] <= -pi:
            angles[i] += tau
        elif angles[i] > pi:
            angles[i] -= tau
    return angles
