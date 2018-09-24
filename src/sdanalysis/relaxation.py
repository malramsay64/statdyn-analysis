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
from typing import NamedTuple

import numpy as np
import pandas
import tables
from scipy.optimize import curve_fit, newton
from scipy.stats import hmean

logger = logging.getLogger(__name__)


class Result(NamedTuple):
    """Hold the result of a relaxation calculation.

    This uses the NamedTuple class to make the access of the returned values more
    transparent and easier to understand.

    """

    mean: float
    error: float


def _msd_function(x: np.ndarray, m: float, b: float) -> np.ndarray:
    """Function to fit the mean squared displacement.

    The relaxation value of the mean squared displacement (MSD) is found by fitting the line

    .. math::
        y = mx + b

    with the relaxation value, i.e. the Diffusion constant, being proportional to :math:`m`.

    """
    return m * x + b


def _exponential_decay(x: np.ndarray, a: float, b: float, c: float = 0) -> np.ndarray:
    """Function used to fit an exponential decay.

    This is is the functional form used to fit a function which exhibits exponential
    decay. The important parameter here is :math:`b`, which is the rate of the decay.

    """
    return a * np.exp(-b * x) + c


# pylint: disable=unused-argument
def _ddx_exponential_decay(
    x: np.ndarray, a: float, b: float, c: float = 0
) -> np.ndarray:
    """The first derivative of the exponential decay function.

    This is the analytical first derivative of the function used for the fit of the
    exponential decay. This is used to speed up the root finding step.

    .. note:

        This function needs to have the same input arguments as the function it is a
        derivative of.

    """
    return -b * a * np.exp(-b * x)


def _d2dx2_exponential_decay(
    x: np.ndarray, a: float, b: float, c: float = 0
) -> np.ndarray:
    """The first derivative of the exponential decay function.

    This is the analytical first derivative of the function used for the fit of the
    exponential decay. This is used to speed up the root finding step.

    .. note:

        This function needs to have the same input arguments as the function it is a
        derivative of.

    """
    return b * b * a * np.exp(-b * x)


# pylint: enable=unused-argument


def diffusion_constant(time: np.ndarray, msd: np.ndarray) -> Result:
    """Compute the diffusion_constant from the mean squared displacement.

    Args:
        time (class:`np.ndarray`): The timesteps corresponding to each msd value.
        msd (class:`np.ndarray`): Values of the mean squared displacement

    Returns:
        diffusion_constant (float): The diffusion constant
        error (float): The error in the fit of the diffusion constant
        (float, float): The diffusion constant

    """
    try:
        linear_region = np.logical_and(msd > 2, msd < 100)
    except FloatingPointError as err:
        logger.exception("%s", err)
        return Result(0, 0)
    try:
        popt, pcov = curve_fit(_msd_function, time[linear_region], msd[linear_region])
    except TypeError as err:
        logger.debug("time: %s", time[linear_region])
        logger.exception("%s", err)
        return Result(0, 0)

    perr = 2 * np.sqrt(np.diag(pcov))
    return Result(popt[0], perr[0])


def threshold_relaxation(
    time: np.ndarray,
    value: np.ndarray,
    threshold: float = 1 / np.exp(1),
    greater: bool = True,
) -> Result:
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
    return Result(time[index], time[index] - time[index - 1])


def exponential_relaxation(
    time: np.ndarray,
    value: np.ndarray,
    sigma: np.ndarray = None,
    value_width: float = 0.3,
) -> Result:
    """Fit a region of the exponential relaxation with an exponential.

    This fits an exponential to the small region around the value 1/e.

    Returns:
        relaxation_time (float): The relaxation time for the given quantity
        error (float): Estimated error of the relaxation time.

    """
    assert time.shape == value.shape
    exp_value = 1 / np.exp(1)
    mask = np.isfinite(value)
    time = time[mask]
    value = value[mask]
    assert np.all(np.isfinite(value))
    fit_region = np.logical_and(
        (exp_value - value_width / 2) < value, (exp_value + value_width / 2) > value
    )
    logger.debug("Num elements: %d", np.sum(fit_region))

    # The number of values has to be greater than the number of fit parameters.
    if np.sum(fit_region) <= 3:
        return Result(np.nan, np.nan)

    zero_est = time[np.argmin(np.abs(value - exp_value))]
    try:
        p0 = (1., 1 / zero_est)
    except (ZeroDivisionError, FloatingPointError) as err:
        logger.exception("%s", err)
        p0 = (1., 0.)

    if sigma is not None:
        sigma = sigma[fit_region]

    try:
        popt, pcov = curve_fit(
            _exponential_decay, time[fit_region], value[fit_region], p0=p0, sigma=sigma
        )
    except RuntimeError as err:
        logger.exception("%s", err)
        return Result(np.nan, np.nan)

    try:
        perr = 2 * np.sqrt(np.diag(pcov))
    except FloatingPointError as err:
        logger.exception("%s", err)
        return Result(np.nan, np.nan)

    logger.debug("Fit Parameters: %s", popt)

    def find_root(a, b):
        return newton(
            _exponential_decay,
            args=(a, b, -exp_value),
            x0=zero_est,
            fprime=_ddx_exponential_decay,
            maxiter=20,
            tol=1e-4,
        )

    try:
        val_mean: float = find_root(*popt)
        try:
            val_min: float = find_root(*(popt - perr))
            val_max: float = find_root(*(popt + perr))
        except FloatingPointError as err:
            logger.exception("%s", err)
            val_min = 0
            val_max = val_mean - val_min
        return Result(val_mean, val_max - val_min)

    except RuntimeError as err:
        logger.exception("%s", err)
        return Result(np.nan, np.nan)


def max_time_relaxation(time: np.ndarray, value: np.ndarray) -> Result:
    """Time at which the maximum value is recorded.

    Args:
        time (np.ndarray): The time index
        value (np.ndarray): The value at each of the time indices

    Returns:
        float: The time at which the maximum value occurs.
        float: Value of the maximum.

    """
    assert time.shape == value.shape
    try:
        max_val_index = np.nanargmax(value)
    except ValueError as err:
        logger.exception("%s", err)
        return Result(np.nan, np.nan)

    if max_val_index == len(value) - 1:
        error = time[max_val_index] - time[max_val_index - 1]
    elif max_val_index == 0:
        error = time[max_val_index + 1] - time[max_val_index]
    else:
        error = (time[max_val_index + 1] - time[max_val_index - 1]) / 2
    return Result(time[max_val_index], error)


def max_value_relaxation(time: np.ndarray, value: np.ndarray) -> Result:
    """Maximum value recorded.

    Args:
        time (np.ndarray): The time index
        value (np.ndarray): The value at each of the time indices

    Returns:
        float: The time at which the maximum value occurs.
        float: Value of the maximum.

    """
    assert time.shape == value.shape
    try:
        max_val_index = np.nanargmax(value)
    except ValueError as err:
        logger.exception("%s", err)
        return Result(np.nan, np.nan)

    if max_val_index == len(value) - 1:
        error = value[max_val_index] - value[max_val_index - 1]
    elif max_val_index == 0:
        error = value[max_val_index] - value[max_val_index + 1]
    else:
        error = (
            (value[max_val_index] - value[max_val_index - 1])
            + (value[max_val_index] - value[max_val_index + 1])
        ) / 2
    return Result(value[max_val_index], error)


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
) -> Result:
    """Compute a single representative value for each dynamic quantity."""
    assert timesteps.shape == values.shape
    if relax_type in ["msd"]:
        return diffusion_constant(timesteps, values)
    if relax_type in ["struct_msd"]:
        return threshold_relaxation(timesteps, values, threshold=0.16, greater=False)
    if relax_type in ["alpha", "gamma"]:
        return max_time_relaxation(timesteps, values)
    return exponential_relaxation(timesteps, values)


def series_relaxation_value(series: pandas.Series) -> float:
    assert series.index.values.shape == series.values.shape
    for level in ["temperature", "pressure"]:
        if level in series.index.names:
            series.reset_index(level=level, drop=True, inplace=True)
    result = compute_relaxation_value(series.index.values, series.values, series.name)
    return result.mean


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
    # Initial frames with any nan value are excluded from analysis. It is assumed they
    # didn't run for a long enough time.
    df_mol = df_mol.dropna()
    df_mol = df_mol.groupby(["temperature", "pressure"]).agg(["mean", hmean])
    df_mol.columns = ["_".join(f) for f in df_mol.columns.tolist()]
    df_mol = df_mol.reset_index()
    relaxations = pandas.concat(relaxation_list)
    pandas.concat([df_mol, relaxations], axis=1).to_hdf(infile, "relaxations")
