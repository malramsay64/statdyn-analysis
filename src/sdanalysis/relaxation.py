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
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas
import tables
from scipy.optimize import curve_fit, newton
from scipy.stats import hmean

logger = logging.getLogger(__name__)


def _msd_function(x: np.ndarray, m: float, b: float) -> np.ndarray:
    return m * x + b


def _exponential_decay(x: np.ndarray, a: float, b: float, c: float = 0) -> np.ndarray:
    return a * np.exp(-b * x) + c


# pylint: disable=unused-argument
def _ddx_exponential_decay(
    x: np.ndarray, a: float, b: float, c: float = 0
) -> np.ndarray:
    return -b * a * np.exp(-b * x)


def _d2dx2_exponential_decay(
    x: np.ndarray, a: float, b: float, c: float = 0
) -> np.ndarray:
    return b * b * a * np.exp(-b * x)


# pylint: enable=unused-argument


def diffusion_constant(time: np.ndarray, msd: np.ndarray) -> Tuple[float, float]:
    """Compute the diffusion_constant from the mean squared displacement.

    Args:
        time (class:`np.ndarray`): The timesteps corresponding to each msd value.
        msd (class:`np.ndarray`): Values of the mean squared displacement

    Returns:
        diffusion_constant (float): The diffusion constant
        error (float): The error in the fit of the diffusion constant
        (float, float): The diffusion constant

    """
    linear_region = np.logical_and(msd > 2, msd < 100)
    try:
        popt, pcov = curve_fit(_msd_function, time[linear_region], msd[linear_region])
    except TypeError:
        return 0, 0

    perr = 2 * np.sqrt(np.diag(pcov))
    return popt[0], perr[0]


def threshold_relaxation(
    time: np.ndarray,
    value: np.ndarray,
    threshold: float = 1 / np.exp(1),
    greater: bool = True,
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
    return time[index], time[index] - time[index - 1]


def exponential_relaxation(
    time: np.ndarray,
    value: np.ndarray,
    sigma: np.ndarray = None,
    value_width: float = 0.3,
) -> Tuple[float, float]:
    """Fit a region of the exponential relaxation with an exponential.

    This fits an exponential to the small region around the value 1/e.

    Returns:
        relaxation_time (float): The relaxation time for the given quantity
        error (float): Estimated error of the relaxation time.

    """
    exp_value = 1 / np.exp(1)
    fit_region = np.logical_and(
        (exp_value - value_width / 2) < value, (exp_value + value_width / 2) > value
    )
    logger.debug("Num elements: %d", np.sum(fit_region))
    zero_est = time[np.argmin(np.abs(value - exp_value))]
    if sigma is not None:
        sigma = sigma[fit_region]
    popt, pcov = curve_fit(
        _exponential_decay,
        time[fit_region],
        value[fit_region],
        p0=[1., 1 / zero_est],
        sigma=sigma,
    )
    perr = 2 * np.sqrt(np.diag(pcov))
    logger.debug("Fit Parameters: %s", popt)

    def find_root(a, b):
        return newton(
            _exponential_decay,
            args=(a, b, -exp_value),
            x0=zero_est,
            fprime=_ddx_exponential_decay,
            maxiter=100,
            tol=1e-4,
        )

    val_mean: float = find_root(*popt)
    val_min: float = find_root(*(popt - perr))
    val_max: float = find_root(*(popt + perr))
    return val_mean, val_max - val_min


def max_time_relaxation(time: np.ndarray, value: np.ndarray) -> Tuple[float, float]:
    """Time at which the maximum value is recorded.

    Args:
        time (np.ndarray): The time index
        value (np.ndarray): The value at each of the time indices

    Returns:
        float: The time at which the maximum value occurs.
        float: Value of the maximum.

    """
    max_val_index = np.nanargmax(value)
    if max_val_index == len(value) - 1:
        error = time[max_val_index] - time[max_val_index - 1]
    elif max_val_index == 0:
        error = time[max_val_index + 1] - time[max_val_index]
    else:
        error = (time[max_val_index + 1] - time[max_val_index - 1]) / 2
    return time[max_val_index], error


# pylint: disable=unused-argument
def max_value_relaxation(time: np.ndarray, value: np.ndarray) -> Tuple[float, float]:
    """Maximum value recorded.

    Args:
        time (np.ndarray): The time index
        value (np.ndarray): The value at each of the time indices

    Returns:
        float: The time at which the maximum value occurs.
        float: Value of the maximum.

    """
    max_val_index = np.nanargmax(value)
    if max_val_index == len(value) - 1:
        error = value[max_val_index] - value[max_val_index - 1]
    elif max_val_index == 0:
        error = value[max_val_index] - value[max_val_index + 1]
    else:
        error = (
            (value[max_val_index] - value[max_val_index - 1])
            + (value[max_val_index] - value[max_val_index + 1])
        ) / 2
    return value[max_val_index], error


def translate_relaxation(quantity: str) -> str:
    translation = {
        "alpha": "max_alpha_time",
        "gamma": "max_gamma_time",
        "com_struct": "tau_F",
        "msd": "diffusion_constant",
        "rot1": "tau_R1",
        "rot2": "tau_R2",
        "struct": "tau_S",
    }
    return translation.get(quantity, quantity)


def compute_relaxation_value(
    timesteps: np.ndarray, values: np.ndarray, relax_type: str
) -> Tuple[float, float]:
    """Compute a single representative value for each dynamic quantity."""
    if relax_type in ["msd"]:
        return diffusion_constant(timesteps, values)
    if relax_type in ["struct_msd"]:
        return threshold_relaxation(timesteps, values, threshold=0.16, greater=False)
    if relax_type in ["alpha", "gamma"]:
        return max_time_relaxation(timesteps, values)
    return exponential_relaxation(timesteps, values)


def series_relaxation_value(series: pandas.Series) -> Tuple[float, float]:
    return compute_relaxation_value(series.index, series.values, series.name)


def compute_relaxations(infile) -> None:
    """Compute the summary time value for the dynamic quantities.

    This computes the characteristic timescale of the dynamic quantities which have been
    calculated and are present in INFILE. The INFILE is a path to the pre-computed
    dynamic quantities and needs to be in the HDF5 format with either the '.hdf5' or
    '.h5' extension.

    The output is written to the table 'relaxations' in INFILE.

    """

    infile = Path(infile)
    # Check is actually an HDF5 file
    try:
        with tables.open_file(str(infile)):
            pass
    except tables.HDF5ExtError:
        raise ValueError("The argument 'infile' requires an hdf5 input file.")

    # Check input file contains the tables required
    with tables.open_file(str(infile)) as src:
        if "/dynamics" not in src:
            raise KeyError(
                "Table 'dynamics' not found in input file,"
                " try rerunning `sdanalysis comp_dynamics`."
            )
        if "/molecular_relaxations" not in src:
            raise KeyError(
                f"Table 'molecular_relaxations' not found in input file,"
                " try rerunning `sdanalysis comp_dynamics`."
            )

    relaxation_list = []
    with pandas.HDFStore(infile) as src:
        for key in src.keys():
            if "dynamics" not in key:
                continue

            df_dyn = src.get(key)
            logger.debug(df_dyn.columns)
            # Remove columns with no relaxation value to calculate
            extra_columns = [
                "mean_displacement",
                "mean_rotation",
                "mfd",
                "overlap",
                "start_index",
            ]
            for col in extra_columns:
                if col in df_dyn.columns:
                    df_dyn.drop(columns=col, inplace=True)

            # Average over all initial times
            df_dyn = df_dyn.groupby(["time", "temperature", "pressure"]).mean()

            relaxations = df_dyn.groupby(["temperature", "pressure"]).agg(
                series_relaxation_value
            )
            relaxations.columns = [
                translate_relaxation(quantity) for quantity in relaxations.columns
            ]
            relaxation_list.append(relaxations)

    df_mol = pandas.read_hdf(infile, "molecular_relaxations")
    df_mol.replace(2 ** 32 - 1, np.nan, inplace=True)
    df_mol.index.names = ["init_frame", "molecule"]
    df_mol = df_mol.groupby(["init_frame", "temperature", "pressure"]).agg(np.mean)
    df_mol = df_mol.groupby(["temperature", "pressure"]).agg(["mean", hmean])
    df_mol.columns = ["_".join(f) for f in df_mol.columns.tolist()]
    relaxations = pandas.concat(relaxation_list)
    pandas.concat([df_mol, relaxations], axis=1).to_hdf(infile, "relaxations")
