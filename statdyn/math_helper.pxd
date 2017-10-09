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


cdef float single_quat_rotation(
        float[:] initial,
        float[:] final
) nogil


cpdef void quaternion_rotation(
        float[:, :] initial,
        float[:, :] final,
        float[:] result
) nogil


cpdef np.ndarray[float, ndim=1] quaternion_angle(
        np.ndarray[float, ndim=2] quat
)


cpdef np.ndarray[float, ndim=2] z2quaternion(
        np.ndarray[float, ndim=1] theta
)


cpdef np.ndarray[float, ndim=1] quaternion2z(
        np.ndarray[float, ndim=2] orientations
)


cpdef void displacement_periodic(
        float[:] box,
        float[:, :] initial,
        float[:, :] final,
        float[:] result
) nogil
