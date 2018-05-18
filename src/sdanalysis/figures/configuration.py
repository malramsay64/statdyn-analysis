#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Plot configuration."""

import logging
from typing import Callable, List

import numpy as np
from bokeh.colors import RGB, Color
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from hsluv import hsluv_to_rgb

from ..frame import Frame, gsdFrame
from ..math_helper import quaternion2z
from ..molecules import Molecule, Trimer
from .trimer import Trimer as TrimerGlyph

logger = logging.getLogger(__name__)


def plotTrimer(mol_plot: figure, source: ColumnDataSource) -> figure:
    """Add the points to a bokeh figure to render the trimer molecule.

    This enables the trimer molecules to be drawn on the figure using only
    the position and the orientations of the central molecule.

    """
    glyph = TrimerGlyph(
        "x",
        "y",
        angle="orientation",
        radius="radius",
        fill_alpha=1,
        fill_color="colour",
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


@np.vectorize
def colour_from_angle(angle: float, saturation: float, luminance: float) -> Color:
    r, g, b = hsluv_to_rgb((angle, saturation, luminance))
    return RGB(r * 256, g * 256, b * 256)


def colour_orientation(orientations: np.ndarray, light_colours=False) -> List[Color]:
    saturation = 85
    luminance = 60
    if light_colours:
        luminance = 85
    return colour_from_angle(
        np.rad2deg(orientations).astype(int), saturation, luminance
    )


def frame2data(
    frame: Frame,
    molecule: Molecule = Trimer(),
    order_function: Callable = None,
    order_list: np.ndarray = None,
):
    angle = quaternion2z(frame.orientation)
    # Colour all particles with the darker shade
    colour = colour_orientation(angle)
    if order_list is not None:
        order_list = np.logical_not(order_list)
        # Colour unordered molecules lighter
        colour[order_list] = colour_orientation(angle, light_colours=True)[order_list]
    elif order_function is not None:
        order = order_function(frame.box, frame.position, frame.orientation)
        if order.dtype in [int, bool]:
            order = order.astype(bool)
        else:
            logger.debug("Order dtype: %s", order.dtype)
            order = order != "liq"
        colour[order] = colour_orientation(angle, light_colours=True)[order]
    data = {
        "x": frame.x_position,
        "y": frame.y_position,
        "orientation": angle,
        "colour": colour,
    }
    return data


def plotFrame(
    frame: Frame,
    order_function: Callable = None,
    order_list: np.ndarray = None,
    source: ColumnDataSource = None,
):
    """Plot snapshot using bokeh."""
    data = frame2data(frame, order_function=order_function, order_list=order_list)
    plot = figure(
        aspect_scale=1,
        match_aspect=True,
        width=920,
        height=800,
        active_scroll="wheel_zoom",
    )
    if source:
        source.data = data
    else:
        source = ColumnDataSource(data=data)
    plotTrimer(plot, source)
    return plot
