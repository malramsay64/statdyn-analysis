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
from libc.math cimport fabs, cos, M_PI, pow, round, sqrt
from libc.limits cimport UINT_MAX
from cython.operator cimport dereference as deref

from .math_helper cimport single_quat_rotation, single_displacement

cdef extern from "<cmath>" namespace "std":
     bint isnan(float x) nogil

cdef extern from "voro++.hh" namespace "voro":
    cdef cppclass container_base:
        pass


    cdef cppclass container:
        container(double,double, double,double, double,double,
                int,int,int, bint,bint,bint, int) except +
        bint compute_cell(voronoicell_neighbor &c, c_loop_all &vl) nogil

        void put(int, double, double, double) nogil
        int total_particles()


    cdef cppclass voronoicell_neighbor:
        voronoicell_neighbor() nogil
        double number_of_faces() nogil

    cdef cppclass c_loop_all:
        c_loop_all(container_base&) nogil
        bint start() nogil
        bint inc() nogil
        int pid() nogil


cpdef np.ndarray[float, ndim=2] _relative_orientations(long [:, :] neighbourlist,
                                                       float[:, :] orientation):
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
    cdef float no_value = -0.
    cdef unsigned int num_mols = neighbourlist.shape[0]
    cdef unsigned int max_neighbours = neighbourlist.shape[1]
    cdef Py_ssize_t mol_index, n, num_neighbours, curr_neighbour

    cdef np.ndarray[float, ndim=2] rel_orient
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


cpdef np.ndarray[float, ndim=1] _orientational_order(long[:, :] neighbourlist,
                                                     float[:, :] orientation,
                                                     float angle_factor=1.):
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
    cdef unsigned int no_value = UINT_MAX
    cdef unsigned int num_mols = orientation.shape[0]
    cdef unsigned int max_neighbours = orientation.shape[1]
    cdef Py_ssize_t mol_index, n, num_neighbours, curr_neighbour

    cdef np.ndarray[float, ndim=1] order_parameter
    cdef double temp_order

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
            order_parameter[mol_index] = <float>temp_order / num_neighbours
        else:
            order_parameter[mol_index] = 0.
        if isnan(order_parameter[mol_index]):
            order_parameter[mol_index] = 0.
    return order_parameter


cpdef np.ndarray[np.uint16_t, ndim=1] compute_voronoi_neighs(float[:] box,
                                                             float[:, :] position) except +:
    cdef unsigned short[:] num_neighs
    cdef Py_ssize_t num_elements = position.shape[0]
    cdef double Lx, Ly, Lz
    cdef int[3] num_blocks
    cdef double N
    num_neighs = np.empty(num_elements, dtype=np.uint16)

    cdef container *configuration
    cdef voronoicell_neighbor *cell
    cdef c_loop_all *voronoi_loop

    Lx = box[0]
    Ly = box[1]
    Lz = box[2]

    # Divide cell into N blocks in each dimension
    # Optimally 5-8 particles per block
    cdef int particles_per_block = 5
    N = sqrt(num_elements/(Lx * Ly * particles_per_block))

    num_blocks[0] = max(<int>round(N * Lx), 1)
    num_blocks[1] = max(<int>round(N * Ly), 1)
    num_blocks[2] = 1

    # Initialize container
    # xmin, xmax,
    # ymin, ymax,
    # zmin, zmax,
    # num_blocks x, y, z
    # periodicity x, y, z
    configuration = new container(
        -Lx/2, Lx/2,
        -Ly/2, Ly/2,
        -Lz/2, Lz/2,
        num_blocks[0], num_blocks[1], num_blocks[2],
        True, True, False,
        num_elements
    )

    with nogil:
        # Add all particles to container
        for i in range(num_elements):
            configuration.put(i, position[i, 0], position[i, 1], position[i, 2])

        # Instance of a class to loop over all particles in configuration
        voronoi_loop = new c_loop_all(<container_base &>deref(configuration))

        # Check constructed properly
        if not voronoi_loop.start():
            # cleanup
            del voronoi_loop, configuration
            with gil:
                raise ValueError("Failed to start loop")

        cell = new voronoicell_neighbor()

        # Loop through all cells
        while True:
            if (configuration.compute_cell(deref(cell), deref(voronoi_loop))):
                num_neighs[voronoi_loop.pid()] = <int>cell.number_of_faces() - 2
            if not voronoi_loop.inc(): break
        # cleanup pointers
        del configuration, cell, voronoi_loop

    return np.asarray(num_neighs)
