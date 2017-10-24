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
from statdyn.math_helper cimport single_quat_rotation, single_displacement


def nn_model():
    from keras.models import load_model
    from pathlib import Path
    return load_model(Path(__file__).parent / 'models/nn-Trimer-model.hdf5')

def dt_model():
    from sklearn.externals import joblib
    from pathlib import Path
    return joblib.load(Path(__file__).parent / 'models/dt-Trimer-model.pkl')

def knn_model():
    from sklearn.externals import joblib
    from pathlib import Path
    return joblib.load(Path(__file__).parent / 'models/knn-Trimer-model.pkl')


cpdef compute_neighbours(np.ndarray[np.float32_t, ndim=1] box,
                         np.ndarray[np.float32_t, ndim=2] position,
                         float max_radius=3.5,
                         int max_neighbours=8):
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

    # The array returned by the nn.getNeighborList command has a lifetime of
    # the nn class object, so will get garbage collected when we leave the
    # scope of this function
    neighs[...] = nn.getNeighborList()[...]
    return neighs


cpdef num_neighbours(np.ndarray[float, ndim=1] box,
                     np.ndarray[float, ndim=2] position,
                     float max_radius=3.5):
    """Compute the number of neighbours for each molecule.

    Args:
        box (np.ndarray[float, ndim=1]): The dimensions of the simulation box
        position (np.ndarray[float, ndim=2]): The positions of all the particles
        max_radius (float): The maximum radius to search for neighbours

    Returns:
        np.ndarray: A list of all the neighbours

    """
    cdef unsigned int max_neighbours = 8
    cdef Py_ssize_t num_mols = position.shape[0]
    cdef unsigned int no_value = UINT_MAX
    cdef unsigned int[:, :] neighbourlist
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


cpdef relative_orientations(np.ndarray[float, ndim=1] box: np.ndarray,
                            np.ndarray[float, ndim=2] position: np.ndarray,
                            np.ndarray[float, ndim=2] orientation: np.ndarray,
                            float max_radius=3.5):
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
    cdef unsigned int max_neighbours = 8
    cdef unsigned int num_mols = position.shape[0]
    cdef Py_ssize_t mol_index, n, num_neighbours, curr_neighbour

    cdef unsigned int[:, :] neighbourlist
    cdef np.ndarray[float, ndim=2] orientations

    neighbourlist = compute_neighbours(box, position, max_radius, max_neighbours)
    orientations = np.zeros((num_mols, max_neighbours), dtype=np.float32)

    for mol_index in range(num_mols):
        num_neighbours = 0
        for n in range(max_neighbours):
            curr_neighbour = neighbourlist[mol_index, n]
            if curr_neighbour < num_mols:
                orientations[mol_index, n] = single_quat_rotation(
                    orientation[curr_neighbour],
                    orientation[mol_index])
            else:
                orientations[mol_index, n] = no_value

    return orientations


cpdef relative_distances(np.ndarray[float, ndim=1] box: np.ndarray,
                         np.ndarray[float, ndim=2] position: np.ndarray,
                         float max_radius=3.5):
    """Compute the relative distance of molecules

    This parameter computed from the relative orientation of the neighbouring
    molecules.

    Args:
        box: Parameters descibing the dimensions of the simulation cell
        position: The positions of all the molecules
        max_radius (float): The maximum radius to search for neighbours

    """
    cdef float no_value = 0.
    cdef unsigned int max_neighbours = 8
    cdef unsigned int num_mols = position.shape[0]
    cdef Py_ssize_t mol_index, n, num_neighbours, curr_neighbour

    cdef unsigned int[:, :] neighbourlist
    cdef np.ndarray[float, ndim=2] distances

    neighbourlist = compute_neighbours(box, position, max_radius, max_neighbours)
    distances = np.zeros((num_mols, max_neighbours), dtype=np.float32)

    for mol_index in range(num_mols):
        num_neighbours = 0
        for n in range(max_neighbours):
            curr_neighbour = neighbourlist[mol_index, n]
            if curr_neighbour < num_mols:
                distances[mol_index, n] = single_displacement(
                    box,
                    position[curr_neighbour],
                    position[mol_index]
                )
            else:
                distances[mol_index, n] = no_value

    return distances


cpdef orientational_order(np.ndarray[float, ndim=1] box: np.ndarray,
                          np.ndarray[float, ndim=2] position: np.ndarray,
                          np.ndarray[float, ndim=2] orientation: np.ndarray,
                          float cutoff=0.8,
                          float max_radius=3.5,
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
    cdef unsigned int max_neighbours = 8
    cdef unsigned int num_mols = position.shape[0]
    cdef Py_ssize_t mol_index, n, num_neighbours, curr_neighbour

    cdef unsigned int[:, :] neighbourlist
    cdef np.ndarray[float, ndim=1] order_parameter

    neighbourlist = compute_neighbours(box, position, max_radius, max_neighbours)
    order_parameter = np.zeros(num_mols, dtype=np.float32)

    for mol_index in range(num_mols):
        num_neighbours = 0
        for n in range(max_neighbours):
            curr_neighbour = neighbourlist[mol_index, n]
            if curr_neighbour < num_mols:
                order_parameter[mol_index] += fabs(cos(
                    angle_factor * single_quat_rotation(orientation[curr_neighbour], orientation[mol_index])
                ))
                num_neighbours += 1
            else:
                break
        if num_neighbours > 1:
            order_parameter[mol_index] /= num_neighbours
        else:
            order_parameter[mol_index] = 0
    return order_parameter > cutoff


def compute_ml_order(
        model,
        np.ndarray[float, ndim=1] box,
        np.ndarray[float, ndim=2] position,
        np.ndarray[float, ndim=2] orientation,
    ):

    cdef float max_radius = 3.5
    cdef unsigned int max_neighbours = 8
    cdef np.ndarray[float, ndim=2] orientations

    orientations = relative_orientations(box, position, orientation, max_radius)
    try:
        return model.predict_classes(orientations)
    except AttributeError:
        return model.predict(orientations)
