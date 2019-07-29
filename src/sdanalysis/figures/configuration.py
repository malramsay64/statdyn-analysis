#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Plot configuration."""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from bokeh import palettes
from bokeh.colors import RGB
from bokeh.models import Circle, ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from hsluv import hpluv_to_rgb

from ..frame import Frame
from ..molecules import Molecule, Trimer
from ..util import orientation2positions, quaternion2z

logger = logging.getLogger(__name__)


def _create_colours(light_colours=False) -> np.ndarray:
    saturation = 100
    luminance = 60
    if light_colours:
        saturation = 80
        luminance = 80
    colours = []
    for hue in range(360):
        r, g, b = hpluv_to_rgb((hue, saturation, luminance))
        colours.append(RGB(r * 256, g * 256, b * 256))
    return np.array(colours)


LIGHT_COLOURS = _create_colours(light_colours=True)
DARK_COLOURS = _create_colours(light_colours=False)


def plot_circles(
    mol_plot: figure,
    source: ColumnDataSource,
    categorical_colour: bool = False,
    factors: Optional[List[Any]] = None,
    colormap=palettes.Category10_10,
) -> figure:
    """Add the points to a bokeh figure to render the trimer molecule.

    This enables the trimer molecules to be drawn on the figure using only
    the position and the orientations of the central molecule.

    """
    glyph_args = dict(
        x="x", y="y", fill_alpha=1, line_alpha=0, radius="radius", fill_color="colour"
    )
    if categorical_colour:
        if factors is None:
            factors = np.unique(source.data["colour"]).astype(str)
        colour_categorical = factor_cmap(
            field_name="colour", factors=factors, palette=colormap
        )
        glyph_args["fill_color"] = colour_categorical

    glyph = Circle(**glyph_args)

    mol_plot.add_glyph(source, glyph=glyph)
    mol_plot.add_tools(
        HoverTool(
            tooltips=[
                ("index", "$index"),
                ("x:", "@x"),
                ("y:", "@y"),
                ("orientation:", "@orientation"),
            ]
        )
    )
    mol_plot.toolbar.active_inspect = None
    return mol_plot


def colour_orientation(orientations: np.ndarray, light_colours=False) -> np.ndarray:
    if light_colours:
        return LIGHT_COLOURS[np.rad2deg(orientations).astype(int)]
    return DARK_COLOURS[np.rad2deg(orientations).astype(int)]


def frame2data(
    frame: Frame,
    order_function: Callable[[Frame], np.ndarray] = None,
    order_list: np.ndarray = None,
    molecule: Molecule = Trimer(),
    categorical_colour: bool = False,
) -> Dict[str, Any]:
    """Convert a Frame to data for plotting in Bokeh.

    This takes a frame and performs all the necessary calculations for plotting, in
    particular the colouring of the orientation and crystal classification.

    Args:
        frame: The configuration which is to be plotted.
        order_function: A function which takes a frame as it's input which can be used
            to classify the crystal.
        order_list: A pre-classified collection of values. This is an alternate
            approach to using the order_function
        molecule: The molecule which is being plotted.
        categorical_colour: Whether to classify as categories, or liquid/crystalline.

    Returns:
        Dictionary containing x, y, colour, orientation and radius values for each
            molecule.

    """
    assert Molecule is not None
    if order_function is not None and order_list is not None:
        raise ValueError("Only one of order_function and order_list can be specified")

    angle = quaternion2z(frame.orientation)

    if categorical_colour:
        if order_function is not None:
            colour = order_function(frame).astype(str)
        elif order_list is not None:
            colour = order_list.astype(str)
        else:
            raise ValueError("No way found to calculate categories.")

    else:
        # Colour all particles with the darker shade
        colour = colour_orientation(angle)
        if order_list is not None:
            order_list = np.logical_not(order_list)
            # Colour unordered molecules lighter
            colour[order_list] = colour_orientation(angle, light_colours=True)[
                order_list
            ]
        elif order_function is not None:
            order = order_function(frame)
            if order.dtype in [int, bool, float]:
                order = np.logical_not(order.astype(bool))
            else:
                logger.debug("Order dtype: %s", order.dtype)
                order = order == "liq"
            logger.debug("Order fraction %.2f", np.mean(order))
            colour[order] = colour_orientation(angle, light_colours=True)[order]

    positions = orientation2positions(molecule, frame.position, frame.orientation)
    data = {
        "x": positions[:, 0],
        "y": positions[:, 1],
        "orientation": np.tile(angle, molecule.num_particles),
        "radius": np.repeat(molecule.get_radii(), len(frame)) * 0.98,
        "colour": np.tile(colour, molecule.num_particles),
    }
    return data


def plot_frame(
    frame: Frame,
    order_function: Optional[Callable[[Frame], np.ndarray]] = None,
    order_list: Optional[np.ndarray] = None,
    source: Optional[ColumnDataSource] = None,
    molecule: Molecule = Trimer(),
    categorical_colour: bool = False,
    factors: Optional[List[Any]] = None,
    colormap=palettes.Category10_10,
):
    """Plot snapshot using bokeh."""
    data = frame2data(
        frame,
        order_function=order_function,
        order_list=order_list,
        molecule=molecule,
        categorical_colour=categorical_colour,
    )
    plot = figure(
        aspect_scale=1,
        match_aspect=True,
        width=920,
        height=800,
        active_scroll="wheel_zoom",
        output_backend="webgl",
    )
    if source:
        source.data = data
    else:
        source = ColumnDataSource(data=data)

    return plot_circles(
        plot, source, categorical_colour, colormap=colormap, factors=factors
    )
