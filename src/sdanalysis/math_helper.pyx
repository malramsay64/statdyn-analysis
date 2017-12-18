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


cpdef np.ndarray[float, ndim=2] rotate_vectors(
        float[:, :] q,
        float[:, :] v):
    """Rotate a series of vectors by a list of quaternions."""
    cdef:
        Py_ssize_t num_quat, num_vect, i, j, res_index
        np.ndarray[float, ndim=2] result
        float w[3]
        float two_over_m = 2

    num_quat = q.shape[0]
    num_vect = v.shape[0]

    result = np.empty((num_vect*num_quat, 3), dtype=np.float32)

    for i in range(num_vect):
        for j in range(num_quat):
            res_index = i*num_quat + j
            w[0] = q[j, 0] * v[i, 0] + q[j, 2]*v[i, 2] - q[j, 3]*v[i, 1];
            w[1] = q[j, 0] * v[i, 1] + q[j, 3]*v[i, 0] - q[j, 1]*v[i, 2];
            w[2] = q[j, 0] * v[i, 2] + q[j, 1]*v[i, 1] - q[j, 2]*v[i, 0];

            result[res_index, 0] = v[i, 0] + two_over_m * (q[j, 2]*w[2] - q[j, 3]*w[1]);
            result[res_index, 1] = v[i, 1] + two_over_m * (q[j, 3]*w[0] - q[j, 1]*w[2]);
            result[res_index, 2] = v[i, 2] + two_over_m * (q[j, 1]*w[1] - q[j, 2]*w[0]);
    return result


cdef void quaternion_rotate_vector(
        float[:] q,
        float[:] v,
        float[:] result) nogil:
    """Rotate a vector by a quaternion

    Code adapted from Moble/Quaternion

    The most efficient formula I know of for rotating a vector by a quaternion is

    v' = v + 2 * r x (s * v + r x v) / m

    where x represents the cross product, s and r are the scalar and vector
    parts of the quaternion, respectively, and m is the sum of the squares of
    the components of the quaternion.  This requires 22 multiplications and 14
    additions, as opposed to 32 and 24 for naive application of `q*v*q.conj()`.
    In this function, I will further reduce the operation count to 18 and 12 by
    skipping the normalization by `m`.  The full version will be implemented in
    another function.

    """
    cdef float w[3]
    cdef float two_over_m = 2

    w[0] = q[0] * v[0] + q[2]*v[2] - q[3]*v[1];
    w[1] = q[0] * v[1] + q[3]*v[0] - q[1]*v[2];
    w[2] = q[0] * v[2] + q[1]*v[1] - q[2]*v[0];

    result[0] = v[0] + two_over_m * (q[2]*w[2] - q[3]*w[1]);
    result[1] = v[1] + two_over_m * (q[3]*w[0] - q[1]*w[2]);
    result[2] = v[2] + two_over_m * (q[1]*w[1] - q[2]*w[0]);


cpdef float single_quat_rotation(
        const float[:, :] orientation,
        int i,
        int j
        ) nogil:
    intermediate = fabs(orientation[i, 0] * orientation[j, 0] +
                   orientation[i, 1] * orientation[j, 1] +
                   orientation[i, 2] * orientation[j, 2] +
                   orientation[i, 3] * orientation[j, 3]
                   )
    if fabs(intermediate - 1.) < QUAT_EPS:
        return 0
    else:
        return <float>2.*acos(intermediate)


cpdef void quaternion_rotation(
        float[:, :] initial,
        float[:, :] final,
        float[:] result) nogil:
    cdef Py_ssize_t nitems = result.shape[0]
    cdef double intermediate

    for i in range(nitems):
        intermediate = fabs(initial[i, 0] * final[i, 0] +
                       initial[i, 1] * final[i, 1] +
                       initial[i, 2] * final[i, 2] +
                       initial[i, 3] * final[i, 3]
                       )
        if fabs(intermediate - 1.) < QUAT_EPS:
            result[i] = 0
        else:
            result[i] = <float>2.*acos(intermediate)


cpdef np.ndarray[float, ndim=1] quaternion_angle(
        float[:, :] quat):
    cdef Py_ssize_t nitems = quat.shape[0]
    cdef np.ndarray[float, ndim=1] result
    result = np.empty(nitems, dtype=np.float32)

    with nogil:
        for i in range(nitems):
            result[i] = 2*acos(quat[i, 0])

    return result



cpdef np.ndarray[float, ndim=2] z2quaternion(
        float[:] theta):
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
        float[:, :] orientations):
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
        float[:] final) nogil:
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
        float[:] result) nogil:
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
