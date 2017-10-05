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
from libc.math cimport fabs, sin, cos, M_PI, acos, sqrt, isnan
from libc.float cimport FLT_EPSILON


cdef float QUAT_EPS = 2*FLT_EPSILON
cdef float M_TAU = 2*M_PI

cdef inline bint close(float a, float b):
    return abs(a-b) <= QUAT_EPS

cpdef float get_quat_eps():
    return QUAT_EPS

cdef float single_quat_rotation(
        float[:] initial,
        float[:] final
):
    return 2.*acos(fabs(
            initial[0] * final[0] +
            initial[1] * final[1] +
            initial[2] * final[2] +
            initial[3] * final[3]
        ))

cpdef np.ndarray[float, ndim=1] quaternion_rotation(
        np.ndarray[float, ndim=2] initial,
        np.ndarray[float, ndim=2] final,
):
    cdef Py_ssize_t nitems = initial.shape[0]
    cdef np.ndarray[float, ndim=1] result
    result = np.empty(nitems, dtype=np.float32)

    for i in range(nitems):
        result[i] = single_quat_rotation(initial[i], final[i])
    return result


cpdef np.ndarray[float, ndim=1] quaternion_angle(
        np.ndarray[float, ndim=2] quat
):
    cdef Py_ssize_t nitems = quat.shape[0]
    cdef np.ndarray[float, ndim=1] result
    result = np.empty(nitems, dtype=np.float32)

    for i in range(nitems):
        result[i] = 2*acos(quat[i, 0])

    return result


cpdef np.ndarray[float, ndim=2] z2quaternion(
        np.ndarray[float, ndim=1] theta
):
    cdef Py_ssize_t i
    cdef Py_ssize_t nitems = theta.shape[0]
    cdef Py_ssize_t w_pos = 0, z_pos = 3
    cdef float angle
    cdef np.ndarray[float, ndim=2] result

    result = np.zeros([nitems, 4], dtype=np.float32)

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

    cdef float q_w
    for i in range(nitems):
        q_w = orientations[i, 0]
        if close(abs(q_w), 1) or close(orientations[i, 3], 0):
            result[i] = 0
        else:
            result[i] = 2.*acos(q_w) / sqrt(1-q_w*q_w) * orientations[i, 3]
            if result[i] > M_PI:
                result[i] -= M_TAU
            if result[i] < -M_PI:
                result[i] += M_TAU

    return result
