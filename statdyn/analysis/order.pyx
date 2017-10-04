# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

"""Perform analysis on the trajectories from the simulation."""

from typing import List, Set

import cython
import numpy as np
from freud.box import Box
from freud.locality import NearestNeighbors

from statdyn.molecules import Molecule, Trimer

from ..molecules import Molecule, Trimer

cimport numpy as np
from libc.math cimport fabs, cos, M_PI
from libc.limits cimport UINT_MAX
from statdyn.math_helper cimport single_quat_rotation

cpdef compute_neighbours(np.ndarray[np.float32_t, ndim=1] box,
                         np.ndarray[np.float32_t, ndim=2] position,
                         float max_radius,
                         int max_neighbours):
    """Compute the neighbour list."""
    ndim = 2
    if ndim == 2:
        simulation_box = Box(box[0], box[1], is2D=True)
    else:
        simulation_box = Box(box[0], box[1], box[2], is2D=False)

    cdef np.ndarray[np.uint32_t, ndim=2] neighs
    neighs = np.empty([position.shape[0], max_neighbours], dtype=np.uint32)

    nn = NearestNeighbors(rmax=max_radius, n_neigh=max_neighbours, strict_cut=True)
    nn.compute(simulation_box, position, position)

    neighs[...] = nn.getNeighborList()[...]
    return neighs


cpdef num_neighbours(np.ndarray[float, ndim=1] box,
                     np.ndarray[float, ndim=2] position,
                     np.ndarray[float, ndim=2] orientation,
                     float max_radius=3.5):
    cdef unsigned int max_neighbours = 8
    cdef Py_ssize_t num_mols = position.shape[0]
    cdef unsigned int no_value = UINT_MAX
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


cpdef orientational_order(np.ndarray[float, ndim=1] box: np.ndarray,
                          np.ndarray[float, ndim=2] position: np.ndarray,
                          np.ndarray[float, ndim=2] orientation: np.ndarray,
                          float max_radius=3.5):

    cdef unsigned int no_value = UINT_MAX
    cdef unsigned int max_neighbours = 8
    cdef unsigned int num_mols = position.shape[0]
    cdef Py_ssize_t mol_index, n, num_neighbours, curr_neighbour

    cdef np.ndarray[unsigned int, ndim=2] neighbourlist
    cdef np.ndarray[float, ndim=1] order_parameter

    neighbourlist = compute_neighbours(box, position, max_radius, max_neighbours)
    order_parameter = np.zeros(num_mols, dtype=np.float32)

    for mol_index in range(num_mols):
        num_neighbours = 0
        for n in range(max_neighbours):
            curr_neighbour = neighbourlist[mol_index, n]
            if curr_neighbour < num_mols:
                order_parameter[mol_index] += fabs(cos(
                    single_quat_rotation(orientation[curr_neighbour], orientation[mol_index])
                ))
                num_neighbours += 1
            else:
                break
        if num_neighbours > 1:
            order_parameter[mol_index] /= num_neighbours
        else:
            order_parameter[mol_index] = 0
    return order_parameter
