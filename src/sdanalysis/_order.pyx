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

import cython
import numpy as np

cimport numpy as np
from libc.math cimport fabs, cos
from libc.limits cimport UINT_MAX

from .math_util cimport single_quat_rotation

cdef extern from "<cmath>" namespace "std":
     bint isnan(float x) nogil

cpdef np.ndarray[float, ndim=2] _relative_orientations(
    const long [:, :] neighbourlist,
    const float[:, :] orientation
):
    """Compute the relative orientations of molecules

    This parameter computed from the relative orientation of the neighbouring
    molecules.

    Args:
        box: Parameters descibing the dimensions of the simulation cell
        position: The positions of all the molecules
        orientation: The orientation of each molecule as a quaternion
        max_radius (float): The maximum radius to search for neighbours
        angle_factor (float): Multiplicative factor for the angle. This allows
            for this function to apply to angles other than 180 deg.

    """
    cdef:
        float no_value = -0.
        unsigned int num_mols = neighbourlist.shape[0]
        unsigned int max_neighbours = neighbourlist.shape[1]
        Py_ssize_t mol_index, n, num_neighbours, curr_neighbour

        np.ndarray[float, ndim=2] rel_orient

    rel_orient = np.empty((num_mols, max_neighbours), dtype=np.float32)

    for mol_index in range(num_mols):
        num_neighbours = 0
        for n in range(max_neighbours):
            curr_neighbour = neighbourlist[mol_index, n]
            if curr_neighbour < num_mols:
                rel_orient[mol_index, n] = single_quat_rotation(
                    orientation,
                    curr_neighbour,
                    mol_index,
                )
            else:
                rel_orient[mol_index, n] = no_value

    return rel_orient


cpdef np.ndarray[float, ndim=1] _orientational_order(
    const long[:, :] neighbourlist,
    const float[:, :] orientation,
    float angle_factor=1.
):
    """Compute the orientational order parameter.

    This parameter computed from the relative orientation of the neighbouring
    molecules.

    Args:
        box: Parameters descibing the dimensions of the simulation cell
        position: The positions of all the molecules
        orientation: The orientation of each molecule as a quaternion
        max_radius (float): The maximum radius to search for neighbours
        angle_factor (float): Multiplicative factor for the angle. This allows
            for this function to apply to angles other than 180 deg.

    """
    cdef:
        unsigned int no_value = UINT_MAX
        unsigned int num_mols = orientation.shape[0]
        unsigned int max_neighbours = orientation.shape[1]
        Py_ssize_t mol_index, n, num_neighbours, curr_neighbour

        np.ndarray[float, ndim=1] order_parameter
        double temp_order

    order_parameter = np.empty(num_mols, dtype=np.float32)

    for mol_index in range(num_mols):
        num_neighbours = 0
        temp_order = 0
        for n in range(max_neighbours):
            curr_neighbour = neighbourlist[mol_index, n]
            if curr_neighbour < num_mols:
                temp_order += fabs(cos(
                    angle_factor *
                    single_quat_rotation(orientation, curr_neighbour, mol_index)
                ))
                num_neighbours += 1
            else:
                break
        if num_neighbours > 1:
            order_parameter[mol_index] = <float>(temp_order / num_neighbours)
        else:
            order_parameter[mol_index] = 0.
        if isnan(order_parameter[mol_index]):
            order_parameter[mol_index] = 0.
    return order_parameter

