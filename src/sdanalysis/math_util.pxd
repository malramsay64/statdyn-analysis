# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""A series of helper functions for performing optimsed maths."""

cimport numpy as np
from libc.math cimport fabs, cos, M_PI, M_2_PI, acos, sqrt


cpdef float single_quat_rotation(
    const float[:, :] orientation,
    const int i,
    const int j
) nogil


cpdef void quaternion_rotation(
    const float[:, :] initial,
    const float[:, :] final,
    float[:] result,
) nogil

cpdef np.ndarray[float, ndim=2] rotate_vectors(
    const float[:, :] quaternions,
    const float[:, :] vectors
)

cdef void quaternion_rotate_vector(
    const float[:] q,
    const float[:] v,
    float[:] result
) nogil

cpdef np.ndarray[float, ndim=1] quaternion_angle(const float[:, :] quat)

cpdef np.ndarray[float, ndim=2] z2quaternion(const float[:] theta)

cpdef np.ndarray[float, ndim=1] quaternion2z(const float[:, :] orientations)

cpdef float single_displacement(
    const float[:] box,
    const float[:] initial,
    const float[:] final
) nogil

cpdef void displacement_periodic(
    const float[:] box,
    const float[:, :] initial,
    const float[:, :] final,
    float[:] result
) nogil
