#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Testing the dynamics module."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from quaternion import (as_float_array, as_quat_array,
                        rotation_intrinsic_distance)

from statdyn.analysis import dynamics

MAX_BOX = 20.


@given(arrays(np.float64, (10, 3), elements=floats(-MAX_BOX/4, MAX_BOX/4)),
       arrays(np.float64, (10, 3), elements=floats(-MAX_BOX/4, MAX_BOX/4)))
def test_mean_sq_displacement(init, final):
    box = np.array([MAX_BOX, MAX_BOX, MAX_BOX])
    result = np.zeros(len(init))
    np_res = np.square(np.linalg.norm(init-final, axis=1))
    dynamics.squaredDisplacment(box, init, final, result)
    assert np.allclose(result, np_res)


@given(arrays(np.float64, (10, 3), elements=floats(-MAX_BOX, -MAX_BOX/2-1e-5)),
       arrays(np.float64, (10, 3), elements=floats(MAX_BOX/2, MAX_BOX)))
def test_mean_sq_displacement_periodicity(init, final):
    box = np.array([MAX_BOX, MAX_BOX, MAX_BOX])
    result = np.zeros(len(init))
    np_res = np.square(np.linalg.norm(init-final, axis=1))
    dynamics.squaredDisplacment(box, init, final, result)
    assert np.all(np.logical_not(np.isclose(result, np_res)))


@given(arrays(np.float64, (1, 4), elements=floats(-1, 1)
              ).filter(lambda x: np.linalg.norm(x) > 0.5),
       arrays(np.float64, (1, 4), elements=floats(-1, 1)
              ).filter(lambda x: np.linalg.norm(x) > 0.5),)
def test_rotationalDisplacement(init, final):
    init_quat = as_quat_array(init)
    final_quat = as_quat_array(final)
    result = np.zeros(len(init))
    dynamics.rotationalDisplacement(init, final, result)
    quat_res = []
    for i, f in zip(init_quat, final_quat):
        quat_res.append(rotation_intrinsic_distance(i.normalized(), f.normalized()))
    assert np.allclose(result, np.array(quat_res), equal_nan=True, atol=5e-2)
