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

import cython
import numpy as np

cimport numpy as np
from libc.math cimport fabs, sin, cos, M_PI, acos, sqrt, isnan, round
from libc.float cimport FLT_EPSILON


cdef float QUAT_EPS = 2*FLT_EPSILON
cdef double M_TAU = 2*M_PI

cdef bint close(float a, float b) nogil:
    return fabs(a-b) <= QUAT_EPS

cpdef float get_quat_eps() nogil:
    return QUAT_EPS

cdef float single_quat_rotation(
        const float[:] initial,
        const float[:] final
) nogil:
    return 2.*acos(fabs(
            initial[0] * final[0] +
            initial[1] * final[1] +
            initial[2] * final[2] +
            initial[3] * final[3]
        ))

cpdef void quaternion_rotation(
        np.ndarray[float, ndim=2] initial,
        np.ndarray[float, ndim=2] final,
        np.ndarray[float, ndim=1] result,
):
    cdef Py_ssize_t nitems = result.shape[0]

    with nogil:
        for i in range(nitems):
            result[i] = 2.*acos(fabs(
                    initial[i, 0] * final[i, 0] +
                    initial[i, 1] * final[i, 1] +
                    initial[i, 2] * final[i, 2] +
                    initial[i, 3] * final[i, 3]
                ))


cpdef np.ndarray[float, ndim=1] quaternion_angle(
        np.ndarray[float, ndim=2] quat
):
    cdef Py_ssize_t nitems = quat.shape[0]
    cdef np.ndarray[float, ndim=1] result
    result = np.empty(nitems, dtype=np.float32)

    with nogil:
        for i in range(nitems):
            result[i] = 2*acos(quat[i, 0])

    return result


cpdef np.ndarray[float, ndim=2] z2quaternion(
        np.ndarray[float, ndim=1] theta
):
    cdef Py_ssize_t i
    cdef Py_ssize_t nitems = theta.shape[0]
    cdef Py_ssize_t w_pos = 0, z_pos = 3
    # Use double for intermediate value in computation
    cdef double angle

    cdef np.ndarray[float, ndim=2] result

    result = np.zeros([nitems, 4], dtype=np.float32)
    with nogil:
        for i in range(nitems):
            angle = theta[i]/2.
            if close(angle, 0):
                result[i, w_pos] = 1.
            else:
                result[i, w_pos] = cos(angle)
                result[i, z_pos] = sin(angle)
    return result


cpdef np.ndarray[float, ndim=1] quaternion2z(
        np.ndarray[float, ndim=2] orientations,
):
    cdef Py_ssize_t nitems = orientations.shape[0]
    cdef np.ndarray[float, ndim=1] result
    result = np.empty(nitems, dtype=np.float32)

    cdef double q_w, q_x, q_y, q_z
    with nogil:
        for i in range(nitems):
            q_w = orientations[i, 0]
            q_x = orientations[i, 1]
            q_y = orientations[i, 2]
            q_z = orientations[i, 3]
            if q_z*q_z == 0.:
                result[i] = 0.
            else:
                result[i] = 2.*acos(q_w) / sqrt(q_x*q_x + q_y*q_y + q_z*q_z) * q_z
                if result[i] > M_PI:
                    result[i] -= M_TAU
    return result


cpdef float single_displacement(
        float[:] box,
        float[:] initial,
        float[:] final
) nogil:
    cdef int j
    cdef double[3] x, inv_box
    cdef double images

    inv_box[0] = 1./box[0]
    inv_box[1] = 1./box[1]
    inv_box[2] = 1./box[2]

    for j in range(3):
        x[j] = initial[j] - final[j]
        if box[j] > FLT_EPSILON:
            images = inv_box[j] * x[j]
            x[j] = box[j] * (images - round(images))

    return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])


cpdef void displacement_periodic(
        float[:] box,
        float[:, :] initial,
        float[:, :] final,
        float[:] result
) nogil:
    cdef int n_elements = result.shape[0]
    cdef int i, j

    # Use doubles for intermediate values of computation
    cdef double[3] x, inv_box
    cdef double images

    inv_box[0] = 1./box[0]
    inv_box[1] = 1./box[1]
    inv_box[2] = 1./box[2]

    for i in range(n_elements):
        for j in range(3):
            x[j] = initial[i, j] - final[i, j]
            if box[j] > FLT_EPSILON:
                images = inv_box[j] * x[j]
                x[j] = box[j] * (images - round(images))

        result[i] = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])
