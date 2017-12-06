#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Plot configuration."""

import logging

import numpy as np
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

from ..math_helper import quaternion2z
from ..molecules import Molecule, Trimer
from .colour import colour_orientation

logger = logging.getLogger(__name__)


def plot_circles(mol_plot, source):
    """Add the points to a bokeh figure to render the trimer molecule.

    This enables the trimer molecules to be drawn on the figure using only
    the position and the orientations of the central molecule.

    """
    mol_plot.circle('x', 'y', radius='radius',
                    fill_alpha=1, fill_color='colour',
                    line_color=None, source=source)
    mol_plot.add_tools(HoverTool(tooltips=[
        ('index', '$index'),
        ('x:', '@x'),
        ('y:', '@y'),
    ]))
    mol_plot.toolbar.active_inspect = None
    return mol_plot


def snapshot2data(snapshot,
                  molecule: Molecule=Trimer(),
                  extra_particles=True,
                  ordering=None,
                  order_list=None,
                  invert_colours=False,
                  ):
    radii = np.ones(snapshot.particles.N)
    orientation = snapshot.particles.orientation
    angle = quaternion2z(orientation)
    position = snapshot.particles.position

    nmols = max(snapshot.particles.body) + 1
    if snapshot.particles.N > nmols:
        orientation = orientation[: nmols]
        angle = angle[: nmols]
        position = position[: nmols]
        radii = radii[: nmols]

    colour = colour_orientation(angle)
    if order_list is not None:
        if invert_colours:
            order_list = np.logical_not(order_list)
        colour[order_list] = colour_orientation(angle, light_colours=True)[order_list]
    elif ordering is not None:
        order = ordering(snapshot.configuration.box, position, orientation)
        if order.dtype in [int, bool]:
            order = order.astype(bool)
        else:
            order = order != 'liq'
        if invert_colours:
            order = np.logical_not(order)
        colour[order] = colour_orientation(angle, light_colours=True)[order]

    if extra_particles:
        position = molecule.orientation2positions(position, orientation)

        logger.debug('Position shape: %s', position.shape)
        radii = np.append([], [radii*r for r in molecule.get_radii()])
        colour = np.tile(colour, molecule.num_particles)

    data = {
        'x': position[:, 0],
        'y': position[:, 1],
        'radius': radii,
        'colour': colour,
    }
    return data


def plot(snapshot, repeat=False, offset=False, order=None,
         extra_particles=True, source=None, order_list=None):
    """Plot snapshot using bokeh."""
    data = snapshot2data(snapshot,
                         molecule=Trimer(),
                         extra_particles=extra_particles,
                         ordering=order,
                         order_list=order_list)
    p = figure(aspect_scale=1, match_aspect=True, width=920, height=800,
               active_scroll='wheel_zoom')
    if source:
        source.data = data
    else:
        source = ColumnDataSource(data=data)
    plot_circles(p, source)
    return p
