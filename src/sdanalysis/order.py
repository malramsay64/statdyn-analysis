#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""
"""

import numpy as np
import rowan
from freud.locality import NearestNeighbors as NearestNeighbours
from freud.voronoi import Voronoi

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

    This parameter computed from the relative orientation of the neighbouring
    molecules.

    Args:
        neighbourlist: The neighbours of each molecule in the simulation
        orientation: The orientation of each molecule as a quaternion
        angle_factor (float): Multiplicative factor for the angle. This allows
            for this function to apply to angles other than 180 deg.

    """
    angles = _neighbour_relative_angle(neighbourlist, orientation)
    return np.cos(angles.mean() * angle_factor)


def compute_ml_order(
    model, box: np.ndarray, position: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    max_radius = 3.5
    max_neighbours = 8
    orientations = relative_orientations(
        box, position, orientation, max_radius, max_neighbours
    )
    return model.predict(orientations)


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
    """Compute the neighbours of each molecule."""
    neighs = setup_neighbours(box, position, max_radius, max_neighbours)
    return neighs.getNeighborList()


def relative_orientations(
    box: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
    max_radius: float = 3.5,
    max_neighbours: int = 8,
) -> np.ndarray:
    neighbours = compute_neighbours(box, position, max_radius, max_neighbours)
    return _neighbour_relative_angle(neighbours, orientation)


def orientational_order(
    box: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
    max_radius: float = 3.5,
    max_neighbours: int = 8,
    order_threshold: float = None,
) -> np.ndarray:
    neighbours = compute_neighbours(box, position, max_radius, max_neighbours)
    if order_threshold is not None:
        return _orientational_order(neighbours, orientation) > order_threshold

    return _orientational_order(neighbours, orientation)


def num_neighbours(
    box: np.ndarray, position: np.ndarray, max_radius: float = 3.5
) -> np.ndarray:
    """Compute the number of neighbours of each molecule."""
    max_neighbours = 9
    neighs = setup_neighbours(box, position, max_radius, max_neighbours)
    return neighs.nlist.neighbor_counts


def relative_distances(
    box: np.ndarray,
    position: np.ndarray,
    max_radius: float = 3.5,
    max_neighbours: int = 8,
) -> np.ndarray:
    """Compute the distance to each neighbour."""
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
