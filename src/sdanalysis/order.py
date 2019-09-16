#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Module for the computation of ordering.

These are tools and utilities for calculating the ordering of local structures.

"""

from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import rowan
from freud.locality import NearestNeighbors as NearestNeighbours
from freud.voronoi import Voronoi

from .frame import Frame
from .util import create_freud_box


def _neighbour_relative_angle(
    neighbourlist: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    num_mols = len(orientation)
    default_vals = np.arange(num_mols).reshape(-1, 1)
    neighbourlist = np.where(neighbourlist < num_mols, neighbourlist, default_vals)
    return rowan.geometry.intrinsic_distance(
        orientation[neighbourlist], orientation.reshape(-1, 1, 4)
    )


def _orientational_order(
    neighbourlist: np.ndarray, orientation: np.ndarray, angle_factor: float = 1.0
) -> np.ndarray:
    """Compute the orientational order parameter.

    This parameter computed from the relative orientation of the neighbouring molecules.
    The orientational order is a number between 0 and 1, with 0 indicating no ordering
    and 1 being perfectly ordered.

    Args:
        neighbourlist: The neighbours of each molecule in the simulation
        orientation: The orientation of each molecule as a quaternion
        angle_factor: Multiplicative factor for the angle. This allows
            for this function to apply to angles other than 180 deg.

    Returns:
        An array with the orientational order of each molecule.

    """
    angles = _neighbour_relative_angle(neighbourlist, orientation)
    return np.square(np.cos(angles.mean() * angle_factor))


def create_ml_ordering(model: Path) -> Callable[[Frame], np.ndarray]:
    """Create a machine learning initialised from a pickled model.

    This reads a machine learning model from a file, creating a function to classify the
    ordering of a configuration.

    Args:
        model: The path to a file containing a pickled model to be loaded using joblib

    Returns:
        A function to classify the ordering within a configuration.

    """
    ml_model = joblib.load(model)

    def compute_ml_order(snap: Frame) -> np.ndarray:
        """Compute the machine learning order of a configuration.

        Args:
            snap: The snapshot for which the ordering should be computed.

        Returns:
            The classification of each molecule in the configuration.

        """
        max_radius = 3.5
        max_neighbours = 8
        orientations = relative_orientations(
            snap.box, snap.position, snap.orientation, max_radius, max_neighbours
        )
        return ml_model.predict(orientations)

    return compute_ml_order


def create_orient_ordering(threshold: float) -> Callable[[Frame], np.ndarray]:
    def compute_orient_ordering(snap: Frame) -> np.ndarray:
        return (
            orientational_order(snap.box, snap.position, snap.orientation) > threshold
        )

    # Set the docstrings based on the construction
    compute_orient_ordering.__doc__ = f"""
        Evaluate ordering of local environments using orientational ordering.

        This evaluates whether each local environment is ordered, with environments containing
        an orientational order parameter greater than {threshold} being considered ordered,
        while values lower are considered disordered.

        Args:
            snap: A frame containing the configuration to be evaluated.

        Returns:
            The evaluation of each local environment. True corresponds to ordered, while False
            is disordered.

        """

    return compute_orient_ordering


def create_neigh_ordering(neighbours: int) -> Callable[[Frame], np.ndarray]:
    def compute_neigh_ordering(snap: Frame) -> np.ndarray:
        return compute_voronoi_neighs(snap.box, snap.position) == neighbours

    # Set the docstrings based on the construction
    compute_neigh_ordering.__doc__ = f"""
        Evaluate ordering of local environments using number of neighbours

        This evaluates whether each local environment is ordered, with environments containing
        {neighbours} neighbours being crystalline, while local environments with more or fewer
        are considered disordered.

        Args:
            snap: A frame containing the configuration to be evaluated.

        Returns:
            The evaluation of each local environment. True corresponds to ordered, while False
            is disordered.

    """

    return compute_neigh_ordering


def setup_neighbours(
    box: np.ndarray,
    position: np.ndarray,
    max_radius: float = 3.5,
    max_neighbours: int = 8,
    is_2D: bool = True,
) -> NearestNeighbours:
    nn = NearestNeighbours(max_radius, max_neighbours, strict_cut=True)
    nn.compute(create_freud_box(box, is_2D), position)
    return nn


def compute_neighbours(
    box: np.ndarray,
    position: np.ndarray,
    max_radius: float = 3.5,
    max_neighbours: int = 8,
) -> np.ndarray:
    """Compute the neighbours of each molecule.

    Args:
        box: The parameters of the simulation cell
        position: The positions of each molecule
        max_radius: The maximum radius to search for neighbours
        max_neighbours: The maximum number of neighbours to find.

    Returns:
        An array containing the index of the neighbours of each molecule. Each molecule
        will have `max_neighbours` listed, with the value `2 ** 32 - 1` indicating a
        missing value.

    """
    neighs = setup_neighbours(box, position, max_radius, max_neighbours)
    return neighs.getNeighborList()


def relative_orientations(
    box: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
    max_radius: float = 3.5,
    max_neighbours: int = 8,
) -> np.ndarray:
    """Find the relative orientations of each neighbouring particle.

    This finds each of the nearest neighbours for each particle and computes the
    orientation of those neighbours relative to the orientation of the central particle.

    Args:
        box: The lengths of the simulation cell in each direction
        position: The position of each particle
        orientation: The orientation of each particle represented as a quaternion
        max_radius: The maximum distance to look to nearest neighbours
        max_neighbours: The maximum number of neighbours considered nearest.

    """
    neighbours = compute_neighbours(box, position, max_radius, max_neighbours)
    return _neighbour_relative_angle(neighbours, orientation)


def orientational_order(
    box: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
    max_radius: float = 3.5,
    max_neighbours: int = 8,
    angle_factor: int = 1,
) -> np.ndarray:
    r"""Compute the orientational order parameter for a given input.

    The orientational order parameter compares the orientation of a particle with that
    of all it's neighbours, using the relation

    ..math:

        \Theta = \sum_{i=1}^N \cos(l(\theta_i - \theta))

    taking the orientation of each of the neighbouring particles compared to the current
    particle.

    Args:
        box: The lengths of the simulation cell in each direction
        position: The position of each particle
        orientation: The orientation of each particle, given as quaternions.
        max_radius: The maximum radius to search for neighbours
        max_neighbours: The maximum number of neighbours to search for
        angle_factor: The factor l in front of each angle. This allows the definition of ordered
            to include alternative orientations. An angle factor of 2 allows for orientations
            at 0 or 180 degrees to be equivalent.
    """
    neighbours = compute_neighbours(box, position, max_radius, max_neighbours)
    return _orientational_order(neighbours, orientation, angle_factor)


def num_neighbours(
    box: np.ndarray, position: np.ndarray, max_radius: float = 3.5
) -> np.ndarray:
    """Calculate the number of neighbours of each molecule.

    This function is optimised to quickly calculate the number of nearest neighbours
    each particle has.

    Args:
        box: The lengths of the simulation cell in each direction
        position: The position of each particle
        max_radius: The maximum radius at which a particle is considered a neighbour.

    """
    max_neighbours = 9
    neighs = setup_neighbours(box, position, max_radius, max_neighbours)
    return neighs.nlist.neighbor_counts


def relative_distances(
    box: np.ndarray,
    position: np.ndarray,
    max_radius: float = 3.5,
    max_neighbours: int = 8,
) -> np.ndarray:
    """Compute the distance to each neighbour.

    Args:
        box: The lengths of the simulation cell in each direction
        position: The position of each particle
        max_radius: The maximum radius at which a particle is considered a neighbour.
        max_neighbours: The maximum number of neighbours to search for

    Returns:
        The distance to each neighbour in a numpy array. Values which correspond
        to missing neighbours are represented by the value -1.

    """
    neighbours = setup_neighbours(box, position, max_radius, max_neighbours)
    distances = np.empty((len(position), max_neighbours))
    distances[:] = neighbours.r_sq_list
    # The distance for neighbours which don't exist is -1. Since this doesn't have a
    # sqrt, replace all values less than 0 with 0.
    distances[distances < 0] = 0
    return np.sqrt(distances)


def compute_voronoi_neighs(box: np.ndarray, position: np.ndarray) -> np.ndarray:
    vor = Voronoi(create_freud_box(box), buff=5)
    nlist = vor.computeNeighbors(position).nlist
    return nlist.neighbor_counts
