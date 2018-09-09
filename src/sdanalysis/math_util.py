#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Utility functions for mathematical operations."""

import numpy as np
import rowan


def quaternion_rotation(initial, final, result):
    result[:] = rowan.geometry.intrinsic_distance(initial, final)


def rotate_vectors(quaternion, vector):
    return rowan.rotate(quaternion, vector)


def quaternion_angle(quaternion) -> np.ndarray:
    return rowan.geometry.angle(quaternion)


def z2quaternion(theta: np.ndarray) -> np.ndarray:
    """Convert a rotation about the z axis to a quaternion.

    This is a helper for 2D simulations, taking the rotation of a particle about the z axis and
    converting it to a quaternion. The input angle `theta` is assumed to be in radians.

    """
    return rowan.from_euler(theta, 0, 0).astype(np.float32)


def quaternion2z(quaternion: np.ndarray) -> np.ndarray:
    """Convert a rotation about the z axis to a quaternion.

    This is a helper for 2D simulations, taking the rotation of a particle about the z axis and
    converting it to a quaternion. The input angle `theta` is assumed to be in radians.

    """
    return rowan.to_euler(quaternion)[:, 0].astype(np.float32)


def displacement_periodic(box, initial, final, result):
    if len(box) > 3 and np.any(box[3:] != 0.):
        raise NotImplementedError(
            "Periodic distances for non-orthorhombic boxes are not yet implemented."
            f"Got xy: {box[3]}, xz: {box[4]}, yz: {box[5]}"
        )
    delta = np.abs(final - initial)
    result[:] = np.linalg.norm(
        np.where(delta > 0.5 * box[:3], delta - box[:3], delta), axis=1
    )
