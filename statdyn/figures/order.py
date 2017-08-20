#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Compute order parameters for figures."""

from typing import List, Set

import numpy as np
from freud.box import Box
from freud.locality import NearestNeighbors
from quaternion import as_quat_array, as_rotation_vector


def get_z_orientation(orientations: np.ndarray) -> np.ndarray:
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
    orientation = as_rotation_vector(as_quat_array(orientations.astype(float)))[:, 2]

    orientation[orientation <= -np.pi] += 2 * np.pi
    orientation[orientation > np.pi] -= 2 * np.pi
    return orientation


def orientational_order(snapshot):
    """Compute the orientational order of a snapshot.

    The orientational order is a number in the range $[0, 1]$ indicating the
    alignment of the neighbouring particles in the parallel or antiparallel
    configuration.

    Molecules which have all the neighbouring molecules aligned either parallel
    or antiparallel will have an orientational order of 1, while molecules with
    every neighbour oriented perpendicular will have an orientational order of
    0.

    Args:
        snapshot: Snapshot of the system

    Returns:
        py:class:`numpy.ndarray`: Array of the orientational order for each
            molecule.

    """
    neighbourlist = compute_neighbours(snapshot)
    nmols = max(snapshot.particles.body) + 1
    angles = get_z_orientation(snapshot.particles.orientation[:nmols])

    order_parameter = np.zeros_like(angles)
    for mol_index, neighbours in enumerate(neighbourlist):
        mol_orientation = angles[mol_index]
        for neighbour in neighbours:
            order_parameter[mol_index] += np.abs(np.cos(
                mol_orientation - angles[neighbour]))
        order_parameter[mol_index] /= len(neighbours)
    return order_parameter


def compute_neighbours(snapshot, max_radius=2.2, max_neighbours=7
                       ) -> List[Set[int]]:
    """Compute the neighbour list for a snapshot."""
    simulation_box = Box(*snapshot.configuration.box,
                         is2D=snapshot.configuration.dimensions == 2)
    nn = NearestNeighbors(rmax=max_radius,
                          n_neigh=max_neighbours,
                          strict_cut=True)
    nn.compute(simulation_box,
               snapshot.particles.position,
               snapshot.particles.position)
    return compute_mol_neighbours(nn.getNeighborList(),
                                  snapshot.particles.body)


def compute_mol_neighbours(neighbourlist, bodies) -> List[Set[int]]:
    """Convert a particle neighbourlist to a molecular neighbourlist."""
    mol_neighbours: List[Set[int]] = []
    nmols = bodies.max()
    for atom_index, neighbours in enumerate(neighbourlist):
        mol_index = bodies[atom_index]
        neighbours = neighbours[neighbours < nmols]
        neighbours = bodies[neighbours]
        neighbours = neighbours[neighbours != mol_index]
        try:
            mol_neighbours[mol_index] |= set(neighbours)
        except IndexError:
            mol_neighbours.append(set(neighbours))
    return mol_neighbours
