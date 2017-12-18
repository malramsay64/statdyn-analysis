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
        float[:, :] orientation,
        int i,
        int j
) nogil


cpdef void quaternion_rotation(
        float[:, :] initial,
        float[:, :] final,
        float[:] result,
) nogil

cpdef np.ndarray[float, ndim=2] rotate_vectors(
        float[:, :] quaternions,
        float[:, :] vectors
)

cdef void quaternion_rotate_vector(
        float[:] q,
        float[:] v,
        float[:] result
) nogil

cpdef np.ndarray[float, ndim=1] quaternion_angle(
        float[:, :] quat
)


cpdef np.ndarray[float, ndim=2] z2quaternion(
        float[:] theta
)


cpdef np.ndarray[float, ndim=1] quaternion2z(
        float[:, :] orientations
)

cpdef float single_displacement(
        float[:] box,
        float[:] initial,
        float[:] final
) nogil

cpdef void displacement_periodic(
        float[:] box,
        float[:, :] initial,
        float[:, :] final,
        float[:] result
) nogil
