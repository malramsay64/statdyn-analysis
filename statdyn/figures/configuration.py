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

from ..analysis.order import num_neighbours, orientational_order
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
                  ordering=False,
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
    if ordering:
        ordered = orientational_order(snapshot.configuration.box,
                                      position,
                                      orientation,
                                      3.5
                                      ) > 0.85
        colour[ordered] = colour_orientation(angle, light_colours=True)[ordered]

    if extra_particles:
        position = molecule.orientation2positions(position, angle)

        logger.debug('Position shape: %s', position.shape)
        radii = np.append([], [radii*r for r in molecule.get_radii()])
        colour = np.append([], [colour]*molecule.num_particles)
    else:
        position = snapshot.particles.position

    data = {
        'x': position[:, 0],
        'y': position[:, 1],
        'radius': radii,
        'colour': colour,
    }
    return data


def plot(snapshot, repeat=False, offset=False, order=False,
         extra_particles=True, source=None):
    """Plot snapshot using bokeh."""
    data = snapshot2data(snapshot,
                         molecule=Trimer(),
                         extra_particles=extra_particles,
                         ordering=order)
    p = figure(aspect_scale=1, match_aspect=True, width=920, height=800,
               active_scroll='wheel_zoom')
    if source:
        source.data = data
    else:
        source = ColumnDataSource(data=data)
    plot_circles(p, source)
    return p
