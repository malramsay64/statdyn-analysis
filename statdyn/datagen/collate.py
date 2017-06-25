#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Series of function to collate thermodynamic data from simulations.

This is a series of functions to collate data from many thermodynamic
properties and then be able to present it in a meaningful manner.
"""

from typing import Callable, List  # pylint: disable=unused-import

import numpy as np
import pandas


def transrot(trans: np.array, rot: np.array) -> float:
    """Compute gamma for a given array or translations and rotations."""
    tmp = np.square(trans).mean() * np.square(rot).mean()
    if tmp:
        return (np.square(trans * np.abs(rot)).mean() - tmp) / (tmp)
    return 0


def allgroups(trans: pandas.DataFrame,
              rot: pandas.DataFrame=None,
              func: Callable[[np.array, np.array], float]=transrot
              ) -> np.array:
    """Apply a function to all times independently."""
    tr_group = trans.groupby('time').displacement
    if rot:
        rot_group = rot.groupby('time').rotation
    else:
        rot_group = trans.groupby('time').displacement
    res = []  # type: List[float]
    for trans, rot in zip(tr_group, rot_group):
        res.append(func(trans[1].values, rot[1].values))
    return np.array(res)
