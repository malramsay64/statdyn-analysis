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
from scipy.stats import spearmanr


def alpha(trans: np.array, rot: np.array) -> float:
    """Compute alpha for a given array or translations and rotations."""
    return (np.power(trans, 4).mean() /
            (2 * (np.square(np.square(trans).mean())))
            ) - 1


def gamma(trans: np.array, rot: np.array) -> float:
    """Compute gamma for a given array or translations and rotations."""
    tmp = np.square(trans).mean() * np.square(rot).mean()
    if tmp:
        return (np.square(trans * np.abs(rot)).mean() - tmp) / (tmp)
    return 0


def spearman_rank(trans: np.array, rot: np.array, fraction: float=0.1) -> float:
    """Compute the Spearman Rank coefficient for fast molecules.

    This takes the molecules with the fastest 10% of the translations or
    rotations and uses this subset to compute the Spearman rank coefficient.
    """
    t_order = np.argsort(trans)
    r_order = np.argsort(np.abs(rot))
    num_elements = int(t_order.shape[0] * fraction)
    argmotion = np.union1d(t_order[:num_elements], r_order[:num_elements])
    rho, phi = spearmanr(trans[argmotion], rot[argmotion])
    return rho


def allgroups(trans: pandas.DataFrame,
              rot: pandas.DataFrame=None,
              func: Callable[[np.array, np.array], float]=gamma
              ) -> np.array:
    """Apply a function to all times independently."""
    tr_group = trans.groupby('time').displacement
    if rot:
        rot_group = rot.groupby('time').rotation
    else:
        rot_group = trans.groupby('time').rotation
    res = []  # type: List[float]
    for trans, rot in zip(tr_group, rot_group):
        res.append(func(trans[1].values, rot[1].values))
    return np.array(res)
