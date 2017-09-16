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

from ..analysis.order import get_z_orientation
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


def snapshot2data(snapshot, molecule: Molecule=Trimer(), extra_particles=True):
    radii = np.ones(snapshot.particles.N)
    orientation = get_z_orientation(snapshot.particles.orientation)
    position = snapshot.particles.position

    nmols = max(snapshot.particles.body) + 1
    if snapshot.particles.N > nmols:
        orientation = orientation[: nmols]
        position = position[: nmols]
        radii = radii[: nmols]

    if extra_particles:
        position = molecule.orientation2positions(
            position,
            orientation,
        )

        logger.debug('Position shape: %s', position.shape)
        radii = np.append([], [radii*r for r in molecule.get_radii()])
    else:
        position = snapshot.particles.position

    data = {
        'x': position[:, 0],
        'y': position[:, 1],
        'radius': radii,
    }
    data['colour'] = colour_orientation(orientation)
    if extra_particles:
        data['colour'] = np.append([], [data['colour']]*molecule.num_particles)
    return data


def plot(snapshot, repeat=False, offset=False, order=False,
         extra_particles=True, source=None):
    """Plot snapshot using bokeh."""
    data = snapshot2data(snapshot, molecule=Trimer(), extra_particles=extra_particles)
    p = figure(aspect_scale=1, match_aspect=True, width=920, height=800,
               active_scroll='wheel_zoom')
    if source:
        source.data=data
    else:
        source = ColumnDataSource(data=data)
    plot_circles(p, source)
    return p
