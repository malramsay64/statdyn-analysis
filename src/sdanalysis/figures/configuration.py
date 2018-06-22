#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Plot configuration."""

import logging
from typing import Any, Callable, Dict, List

import numpy as np
from bokeh.colors import RGB, Color
from bokeh.models import Circle, ColumnDataSource, HoverTool
from bokeh.plotting import figure
from hsluv import hpluv_to_rgb

from ..frame import Frame, HoomdFrame
from ..math_util import quaternion2z
from ..molecules import Molecule, Trimer
from .trimer import Trimer as TrimerGlyph

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


def plot_trimer(mol_plot: figure, source: ColumnDataSource) -> figure:
    """Add the points to a bokeh figure to render the trimer molecule.

    This enables the trimer molecules to be drawn on the figure using only
    the position and the orientations of the central molecule.

    """
    glyph = TrimerGlyph(
        x="x", y="y", angle="orientation", fill_alpha=1, fill_color="colour"
    )

    mol_plot.add_glyph(source, glyph=glyph)
    mol_plot.add_tools(
        HoverTool(
            tooltips=[
                ("index", "$index"),
                ("x:", "@x"),
                ("y:", "@y"),
                ("orientation:", "@angle"),
            ]
        )
    )
    mol_plot.toolbar.active_inspect = None
    return mol_plot


def plot_circles(mol_plot: figure, source: ColumnDataSource) -> figure:
    """Add the points to a bokeh figure to render the trimer molecule.

    This enables the trimer molecules to be drawn on the figure using only
    the position and the orientations of the central molecule.

    """
    glyph = Circle(
        x="x", y="y", fill_alpha=1, line_alpha=0, radius="radius", fill_color="colour"
    )

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
    order_function: Callable = None,
    order_list: np.ndarray = None,
    molecule: Molecule = None,
) -> Dict[str, Any]:
    angle = quaternion2z(frame.orientation)
    # Colour all particles with the darker shade
    colour: np.ndarray = colour_orientation(angle)
    if order_list is not None:
        order_list = np.logical_not(order_list)
        # Colour unordered molecules lighter
        colour[order_list] = colour_orientation(angle, light_colours=True)[order_list]
    elif order_function is not None:
        order = order_function(frame.box, frame.position, frame.orientation)
        if order.dtype in [int, bool]:
            order = np.logical_not(order.astype(bool))
        else:
            logger.debug("Order dtype: %s", order.dtype)
            order = order == "liq"
        logger.debug("Order fraction %.2f", np.mean(order))
        colour[order] = colour_orientation(angle, light_colours=True)[order]
    if molecule is not None:
        positions = molecule.orientation2positions(frame.position, frame.orientation)
        data = {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "orientation": np.tile(angle, molecule.num_particles),
            "radius": np.repeat(molecule.get_radii(), len(frame)) * 0.98,
            "colour": np.tile(colour, molecule.num_particles),
        }
    else:
        data = {
            "x": frame.x_position,
            "y": frame.y_position,
            "orientation": angle,
            "colour": colour,
        }
    return data


def plot_frame(
    frame: Frame,
    order_function: Callable = None,
    order_list: np.ndarray = None,
    source: ColumnDataSource = None,
    molecule: Molecule = None,
):
    """Plot snapshot using bokeh."""
    data = frame2data(
        frame, order_function=order_function, order_list=order_list, molecule=molecule
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
    if molecule is not None:
        plot_circles(plot, source)
    else:
        plot_trimer(plot, source)
    return plot
