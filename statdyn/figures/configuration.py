#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Plot configuration."""

import numpy as np
from bokeh.plotting import figure

from ..analysis.order import get_z_orientation, orientational_order
from .colour import colour_orientation


def trimer_figure(mol_plot, xpos, ypos, orientations, mol_colours,
                  extra_particles=True):
    """Add the points to a bokeh figure to render the trimer molecule.

    This enables the trimer molecules to be drawn on the figure using only
    the position and the orientations of the central molecule.

    """
    mol_plot.circle(xpos, ypos, radius=1,
                    fill_alpha=1, color=mol_colours,
                    line_color=None)
    if extra_particles:
        atom1_x = xpos - np.sin(orientations - np.pi/3)
        atom1_y = ypos + np.cos(orientations - np.pi/3)
        mol_plot.circle(atom1_x, atom1_y, radius=0.64,
                        fill_alpha=1, color=mol_colours,
                        line_color=None)
        atom2_x = xpos - np.sin(orientations + np.pi/3)
        atom2_y = ypos + np.cos(orientations + np.pi/3)
        mol_plot.circle(atom2_x, atom2_y, radius=0.64,
                        fill_alpha=1, color=mol_colours,
                        line_color=None)
    return mol_plot


def plot(snapshot, repeat=False, offset=False, order=False, extra_particles=True):
    """Plot snapshot using bokeh."""
    try:
        Lx, Ly = snapshot.configuration.box[:2]
    except AttributeError:
        Lx, Ly = snapshot.box.Lx, snapshot.box.Ly

    plot_range = (-Ly/2, Ly/2)
    nmols = np.max(snapshot.particles.body) + 1
    x = snapshot.particles.position[:nmols, 0]
    y = snapshot.particles.position[:nmols, 1]
    if offset:
        x = x % Lx
        y = y % Ly
        plot_range = (0, Ly)

    orientations = get_z_orientation(snapshot.particles.orientation[:nmols])
    mol_colours = colour_orientation(orientations)

    if order:
        order = orientational_order(snapshot)
        mol_colours[order < 0.9] = colour_orientation(
            orientations[order < 0.9], light_colours=True)

    p = figure(x_range=plot_range, y_range=plot_range,
               active_scroll='wheel_zoom', width=800, height=800)
    trimer_figure(p, x, y, orientations, mol_colours, extra_particles=extra_particles)
    if repeat:
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            dx *= Lx
            dy *= Ly
            trimer_figure(p, x+dx, y+dy, orientations, mol_colours)
            trimer_figure(p, x-dx, y-dy, orientations, mol_colours)
    return p
