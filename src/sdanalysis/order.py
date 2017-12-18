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
from scipy.spatial import cKDTree

from ._order import (_orientational_order, _relative_orientations,
                     compute_voronoi_neighs)


def dt_model():
    from sklearn.externals import joblib
    from pathlib import Path
    return joblib.load(Path(__file__).parent / 'models/dt-Trimer-model.pkl')


def knn_model():
    from sklearn.externals import joblib
    from pathlib import Path
    return joblib.load(Path(__file__).parent / 'models/knn-Trimer-model.pkl')


def compute_ml_order(model,
                     box: np.ndarray,
                     position: np.ndarray,
                     orientation: np.ndarray) -> np.ndarray:
    max_radius = 3.5
    max_neighbours = 8

    orientations = relative_orientations(
        box,
        position,
        orientation,
        max_radius,
        max_neighbours,
    )
    return model.predict(orientations)


def compute_neighbour_tree(box: np.ndarray,
                           position: np.ndarray) -> cKDTree:
    """Initialise a cKDTree to compute neighbours from."""
    pos_position = position + box[:3]/2
    return cKDTree(pos_position, boxsize=box[:3])


def compute_neighbours(box: np.ndarray,
                       position: np.ndarray,
                       max_radius: float=3.5,
                       max_neighbours: int=8) -> np.ndarray:
    """Compute the neighbours of each molecule."""
    neigh_tree = compute_neighbour_tree(box, position)
    return neigh_tree.query(neigh_tree.data,
                            max_neighbours+1,
                            distance_upper_bound=max_radius,
                            n_jobs=-1)[1][:, 1:]


def relative_orientations(box: np.ndarray,
                          position: np.ndarray,
                          orientation: np.ndarray,
                          max_radius: float=3.5,
                          max_neighbours: int=8) -> np.ndarray:
    neighbours = compute_neighbours(box, position, max_radius, max_neighbours)
    return _relative_orientations(neighbours, orientation)


def orientational_order(box: np.ndarray,
                        position: np.ndarray,
                        orientation: np.ndarray,
                        max_radius: float=3.5,
                        max_neighbours: int=8) -> np.ndarray:
    neighbours = compute_neighbours(box, position, max_radius, max_neighbours)
    return _orientational_order(neighbours, orientation)


def num_neighbours(box: np.ndarray,
                   position: np.ndarray,
                   max_radius: float=3.5) -> np.ndarray:
    """Compute the number of neighbours of each molecule."""
    max_neighbours = 8
    dist = relative_distances(box, position, max_radius, max_neighbours)
    return max_neighbours - np.isinf(dist).sum(axis=1)


def relative_distances(box: np.ndarray,
                       position: np.ndarray,
                       max_radius: float=3.5,
                       max_neighbours: int=8) -> np.ndarray:
    """Compute the distance to each neighbour."""
    neigh_tree = compute_neighbour_tree(box, position)
    return neigh_tree.query(neigh_tree.data,
                            max_neighbours+1,
                            distance_upper_bound=max_radius,
                            n_jobs=-1)[0][:, 1:]
