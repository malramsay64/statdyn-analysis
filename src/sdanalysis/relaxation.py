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
        linear_region = np.logical_and(2 < msd, msd < 50)
        if np.sum(linear_region) <= 2:
            return Result(np.nan, np.nan)
    except FloatingPointError as err:
        logger.exception("%s", err)
        return Result(np.nan, np.nan)
    try:
        popt, pcov = curve_fit(_msd_function, time[linear_region], msd[linear_region])
    except TypeError as err:
        logger.debug("time: %s", time[linear_region])
        logger.exception("%s", err)
        return Result(np.nan, np.nan)

    perr = 2 * np.sqrt(np.diag(pcov))
    return Result(popt[0], perr[0])


def threshold_relaxation(
    time: np.ndarray,
    value: np.ndarray,
    threshold: float = 1 / np.exp(1),
    decay: bool = True,
) -> Result:
    """Compute the relaxation through the reaching of a specific value.

    Args:
        time (class:`np.ndarray`): The timesteps corresponding to each msd value.
        value (class:`np.ndarray`): Values of the relaxation paramter

    Returns:
        relaxation time (float): The relaxation time for the given quantity.
        error (float): The error in the fit of the relaxation

    """
    try:
        if decay:
            index = np.argmax(value < threshold)
        else:
            index = np.argmax(value > threshold)
    except FloatingPointError:
        return Result(np.nan, np.nan)
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
    if time.shape != value.shape:
        raise RuntimeError(
            "Time and value have different shapes. "
            "time: {time.shape}, values: {value.shape}"
        )

    exp_value = 1 / np.exp(1)
    mask = np.isfinite(value)
    time = time[mask]
    value = value[mask]
    if np.any(~np.isfinite(value)):
        raise ValueError("There are non-finite values present in `value`.")
    fit_region = np.logical_and(
        (exp_value - value_width / 2) < value, (exp_value + value_width / 2) > value
    )
    logger.debug("Num elements: %d", np.sum(fit_region))

    # The number of values has to be greater than the number of fit parameters.
    if np.sum(fit_region) <= 3:
        return Result(np.nan, np.nan)

    zero_est = time[np.argmin(np.abs(value - exp_value))]
    try:
        p0 = (1.0, 1 / zero_est)
    except (ZeroDivisionError, FloatingPointError) as err:
        logger.warning("Handled exception in estimating zero\n%s", err)
        p0 = (1.0, 0.0)

    if sigma is not None:
        sigma = sigma[fit_region]

    try:
        popt, pcov = curve_fit(
            _exponential_decay, time[fit_region], value[fit_region], p0=p0, sigma=sigma
        )
    except (RuntimeError, FloatingPointError) as err:
        logger.warning("Exception in fitting curve, returning Nan\n%s", err)
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
            logger.warning("Handled Exception in calculating Error bars\n%s", err)
            val_min = 0
            val_max = val_mean - val_min
        if val_mean > 0:
            return Result(val_mean, val_max - val_min)
        logger.warning("Rate is less than 0, returning Nan")
        return Result(np.nan, np.nan)

    except RuntimeError as err:
        logger.warning("Failed to converge on value, returning NaN")
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
    if time.shape != value.shape:
        raise RuntimeError(
            "Time and value have different shapes. "
            "time: {time.shape}, values: {value.shape}"
        )

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
        time: The time index
        value: The value at each of the time indices

    Returns:
        float: The time at which the maximum value occurs.
        float: Value of the maximum.

    """
    if time.shape != value.shape:
        raise RuntimeError(
            "Time and value have different shapes. "
            "time: {time.shape}, values: {value.shape}"
        )

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
    """Compute a single representative value for each dynamic quantity.

    Args:
        timesteps: The timestep for each value of the relaxation.
        values: The values of the relaxation quantity for each time interval.
        relax_type: A string describing the relaxation.

    Returns:
        The representative relaxation time for a quantity.


    There are some special values of the relaxation which are treated in a special way.
    The main one of these is the "msd", for which the relaxation is fitted to a straight
    line. The "struct_msd" relaxation, is a threshold_relaxation, with the time required
    to pass the threshold of 0.16. The other relaxations which are treated separately
    are the "alpha" and "gamma" relaxations, where the relaxation time is the maximum of
    these functions.

    All other relaxations are assumed to have the behaviour of exponential decay, with
    the representative time being how long it takes to decay to the value 1/e.

    """
    if timesteps.shape != values.shape:
        raise RuntimeError(
            "Timesteps and values have different shapes. "
            "timesteps: {timesteps.shape}, values: {values.shape}"
        )

    if relax_type in ["msd"]:
        return diffusion_constant(timesteps, values)
    if relax_type in ["struct_msd"]:
        return threshold_relaxation(timesteps, values, threshold=0.16, decay=False)
    if relax_type in ["alpha", "gamma"]:
        return max_time_relaxation(timesteps, values)
    return threshold_relaxation(timesteps, values)


def series_relaxation_value(series: pandas.Series) -> float:
    """This is a utility function for calculating the relaxation of a pandas Series.

    When a `pandas.Series` object, which has an index being the timesteps, and the name
    of the series being the dynamic quantity, this function provides a simple method of
    calculating the relaxation aggregation. In particular this function is useful to use
    with the aggregate function.

    Args:
        series: The series containing the relaxation quantities

    Returns:
        The calculated value of the relaxation.


    .. note:

        This function will discard the error in the relaxation calculation for
        simplicity in working with the resulting DataFrame.

    """
    if series.index.values.shape != series.values.shape:
        raise RuntimeError(
            "Index and values have different shapes."
            f"index: {series.index.value.shape}, values: {series.value.shape}"
        )

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

    with pandas.HDFStore(infile) as src:
        df_dyn = src.get("dynamics")

    logger.debug(df_dyn.columns)
    # Remove columns with no relaxation value to calculate
    extra_columns = ["mean_displacement", "mean_rotation", "mfd", "overlap"]
    for col in extra_columns:
        if col in df_dyn.columns:
            df_dyn.drop(columns=col, inplace=True)

    relaxations = df_dyn.groupby(["keyframe", "temperature", "pressure"]).agg(
        series_relaxation_value
    )

    logger.info("Shape of df_dyn relaxations is %s", df_dyn.shape)

    with pandas.HDFStore(infile) as src:
        df_mol_input = src.get("molecular_relaxations")

    logger.info("Shape of df_mol_input relaxations is %s", df_mol_input.shape)

    df_mol = compute_molecular_relaxations(
        pandas.read_hdf(infile, "molecular_relaxations")
    )
    logger.info("Shape of df_mol relaxations is %s", df_mol.shape)

    df_all = df_mol.join(
        relaxations, on=["keyframe", "temperature", "pressure"], lsuffix="mol"
    ).reset_index()
    logger.info("Shape of all relaxations is %s", df_all.shape)

    if "temperature" not in df_all.columns:
        raise RuntimeError(
            "Temperature not in columns, something has gone really wrong."
        )

    if "pressure" not in df_all.columns:
        raise RuntimeError("Pressure not in columns, something has gone really wrong.")

    df_all.to_hdf(infile, "relaxations")


def compute_molecular_relaxations(df: pandas.DataFrame) -> pandas.DataFrame:
    if "temperature" not in df.columns:
        raise ValueError("The column 'temperature' is required")

    if "pressure" not in df.columns:
        raise ValueError("The column 'pressure' is required")

    if "keyframe" not in df.columns:
        raise ValueError("The column 'keyframe' is required")

    logger.debug("Initial molecular shape: %s", df.shape)
    df.replace(2 ** 32 - 1, np.nan, inplace=True)
    # Initial frames with any NaN value are excluded from analysis. It is assumed they
    # didn't run for a long enough time.
    df = df.groupby(["keyframe", "temperature", "pressure"]).filter(
        lambda x: x.isna().sum().sum() == 0
    )
    logger.debug("Filtered molecular shape: %s", df.shape)

    # Calculate statistics for each initial_frame
    df = df.groupby(["keyframe", "temperature", "pressure"]).agg(["mean", hmean])

    logger.debug("Aggregated molecular shape: %s", df.shape)
    df.columns = ["_".join(f) for f in df.columns.tolist()]
    return df
