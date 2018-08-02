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
from freud.box import Box
from freud.locality import NearestNeighbors as NearestNeighbours
from freud.voronoi import Voronoi

from ._order import _orientational_order, _relative_orientations


def dt_model():
    from sklearn.externals import joblib
    from pathlib import Path

    return joblib.load(Path(__file__).parent / "models/dt-Trimer-model.pkl")


def knn_model():
    from sklearn.externals import joblib
    from pathlib import Path

    return joblib.load(Path(__file__).parent / "models/knn-Trimer-model.pkl")


def compute_ml_order(
    model, box: np.ndarray, position: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    max_radius = 3.5
    max_neighbours = 8
    orientations = relative_orientations(
        box, position, orientation, max_radius, max_neighbours
    )
    return model.predict(orientations)


def create_freud_box(box: np.ndarray, is_2D=True) -> Box:
    # pylint: disable=invalid-name
    Lx, Ly, Lz = box[:3]
    xy = xz = yz = 0
    if len(box) == 6:
        xy, xz, yz = box[3:6]
    if is_2D:
        return Box(Lx=Lx, Ly=Ly, xy=xy, is2D=is_2D)
    return Box(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz)
    # pylint: disable=invalid-name


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
    return neighs.getNeighborList().astype(np.long)


def relative_orientations(
    box: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
    max_radius: float = 3.5,
    max_neighbours: int = 8,
) -> np.ndarray:
    neighbours = compute_neighbours(box, position, max_radius, max_neighbours)
    return _relative_orientations(neighbours, orientation)


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
    return np.sqrt(neighbours.getRsqList())


def compute_voronoi_neighs(box: np.ndarray, position: np.ndarray) -> np.ndarray:
    vor = Voronoi(create_freud_box(box), buff=5)
    nlist = vor.computeNeighbors(position).getNeighborList()
    return nlist.neighbor_counts
