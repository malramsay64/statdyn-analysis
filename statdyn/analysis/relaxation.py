#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""These are a series of summary values of the dynamics quantities.

This provides methods of easily comparing values across variables.
"""

import logging
from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit, newton

logger = logging.getLogger(__name__)


def _msd_function(x: np.ndarray, m: float, b: float) -> np.ndarray:
    return m*x + b


def _exponential_decay(x: np.ndarray, a: float, b: float, c: float=0) -> np.ndarray:
    return a * np.exp(-b * x) + c


def _ddx_exponential_decay(x: np.ndarray, a: float, b: float, c: float=0) -> np.ndarray:
    return -b * a * np.exp(-b * x)


def _d2dx2_exponential_decay(x: np.ndarray, a: float, b: float, c: float=0) -> np.ndarray:
    return b * b * a * np.exp(-b * x)


def diffusion_constant(time: np.ndarray,
                       msd: np.ndarray,
                       sigma: np.ndarray=None,
                       ) -> Tuple[float, float]:
    """Compute the diffusion_constant from the mean squared displacement.

    Args:
        time (class:`np.ndarray`): The timesteps corresponding to each msd value.
        msd (class:`np.ndarray`): Values of the mean squared displacement

    Returns:
        diffusion_constant (float): The diffusion constant
        error (float): The error in the fit of the diffusion constant
        (float, float): The diffusion constant

    """
    linear_region = np.logical_and(2 < msd, msd < 100)
    try:
        popt, pcov = curve_fit(_msd_function, time[linear_region], msd[linear_region])
    except TypeError:
        return 0, 0
    perr = 2*np.sqrt(np.diag(pcov))
    return popt[0], perr[0]


def threshold_relaxation(time: np.ndarray,
                         value: np.ndarray,
                         threshold: float=1/np.exp(1),
                         greater: bool=True,
                         ) -> Tuple[float, float]:
    """Compute the relaxation through the reaching of a specific value.

    Args:
        time (class:`np.ndarray`): The timesteps corresponding to each msd value.
        value (class:`np.ndarray`): Values of the relaxation paramter

    Returns:
        relaxation time (float): The relaxation time for the given quantity.
        error (float): The error in the fit of the relaxation

    """
    if greater:
        index = np.argmax(value > threshold)
    else:
        index = np.argmin(value < threshold)
    return time[index], time[index]-time[index-1]


def exponential_relaxation(time: np.ndarray,
                           value: np.ndarray,
                           sigma: np.ndarray=None,
                           value_width: float=0.3) -> Tuple[float, float, float]:
    """Fit a region of the exponential relaxation with an exponential.

    This fits an exponential to the small region around the value 1/e.

    Returns:
        relaxation_time (float): The relaxation time for the given quantity
        error_min (float): The minmum error value
        error_max (float): The maximum error value

    """
    exp_value = 1/np.exp(1)
    fit_region = np.logical_and((exp_value - value_width/2) < value,
                                (exp_value + value_width/2) > value)
    logger.debug('Num elements: %d', np.sum(fit_region))
    zero_est = time[np.argmin(np.abs(value - exp_value))]
    if sigma is not None:
        sigma = sigma[fit_region]
    popt, pcov = curve_fit(
        _exponential_decay,
        time[fit_region],
        value[fit_region],
        p0=[1., 1/zero_est],
        sigma=sigma,
    )
    perr = 2*np.sqrt(np.diag(pcov))
    logger.debug('Fit Parameters: %s', popt)

    def find_root(a, b):
        return newton(
            _exponential_decay,
            args=(a, b, -exp_value),
            x0=zero_est,
            fprime=_ddx_exponential_decay,
            maxiter=100,
            tol=1e-4
        )

    val_mean: float = find_root(*popt)
    val_min: float = find_root(*(popt-perr))
    val_max: float = find_root(*(popt+perr))
    return val_mean, val_mean - val_min, val_max - val_min
