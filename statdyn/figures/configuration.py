#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Plot configuration."""

from bokeh.plotting import figure

from .colour import clean_orientation, colour_orientation


def plot(snapshot, repeat=False, offset=False):
    """Plot snapshot using bokeh."""
    try:
        Lx, Ly = snapshot.configuration.box[:2]
    except AttributeError:
        Lx, Ly = snapshot.box.Lx, snapshot.box.Ly

    plot_range = (-Ly/2, Ly/2)
    x = snapshot.particles.position[:, 0]
    y = snapshot.particles.position[:, 1]
    if offset:
        x = x % Lx
        y = y % Ly
        plot_range = (0, Ly)

    radius = (snapshot.particles.typeid * -0.362444) + 1
    orientation = colour_orientation(clean_orientation(snapshot))

    p = figure(x_range=plot_range, y_range=plot_range,
               active_scroll='wheel_zoom', width=800, height=800,
               output_backend='webgl')
    p.scatter(x, y, radius=radius*0.95,
              fill_alpha=1, color=orientation,
              line_color=None)
    if repeat:
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            dx *= Lx
            dy *= Ly
            p.scatter(x+dx, y+dy, radius=radius*0.95,
                      fill_alpha=1, color=orientation,
                      line_color=None)
            p.scatter(x-dx, y-dy, radius=radius*0.95,
                      fill_alpha=1, color=orientation,
                      line_color=None)
    return p
