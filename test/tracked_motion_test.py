#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Ensure the tracked motion class is working as intended."""

import freud
import numpy as np
import rowan
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, lists

from sdanalysis.dynamics import TrackedMotion


def test_simple_orientation():
    box = freud.box.Box(10, 10, 10)
    motion = TrackedMotion(box, np.zeros((1, 3)), rowan.from_euler([0], [0], [0]))
    for i in range(10):
        motion.add(np.zeros((1, 3)), rowan.from_euler([np.pi / 4 * (i + 1)], [0], [0]))

    assert np.isclose(np.linalg.norm(motion.delta_rotation), np.pi / 4 * 10)


@given(lists(floats(-np.pi / 2, np.pi / 2), min_size=1))
def test_orientation(rotations):
    box = freud.box.Box(10, 10, 10)
    motion = TrackedMotion(box, np.zeros((1, 3)), rowan.from_euler([0], [0], [0]))
    orientations = np.cumsum(rotations)
    for orientation in orientations:
        motion.add(np.zeros((1, 3)), rowan.from_euler([orientation], [0], [0]))

    print(sum(rotations))
    print(motion.delta_rotation.sum())
    assert np.isclose(motion.delta_rotation[0, 0], np.sum(rotations), atol=1e-8)


def test_simple_translation():
    box = freud.box.Box(10, 10, 10)
    motion = TrackedMotion(box, np.zeros((1, 3)), rowan.from_euler([0], [0], [0]))
    for i in range(100):
        motion.add(np.ones((1, 3)) * (i + 1), rowan.from_euler([0], [0], [0]))

    assert np.isclose(np.linalg.norm(motion.delta_translation), np.sqrt(3) * 100)


@given(arrays(np.float64, (10, 3), floats(-4, 4)))
def test_translation(translations):
    box = freud.box.Box(10, 10, 10)
    motion = TrackedMotion(box, np.zeros((1, 3)), rowan.from_euler([0], [0], [0]))
    positions = np.cumsum(translations, axis=0)
    for position in positions:
        motion.add(position, rowan.from_euler([0], [0], [0]))

    print(positions)
    print(motion.delta_translation)
    assert np.allclose(motion.delta_translation, positions[-1], atol=1e-7)
