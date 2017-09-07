#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Create an interactive view of a configuration."""

import logging
from pathlib import Path

import gsd.hoomd
from bokeh.layouts import row, widgetbox
from bokeh.models import Button, ColumnDataSource, Select, Slider, TextInput
from bokeh.plotting import curdoc, figure

from statdyn.figures.configuration import plot_circles, snapshot2data
from statdyn.molecule import Trimer

logger = logging.getLogger(__name__)

# Definition of initial state
trj = None
snapshot = None
extra_particles = True
molecule = Trimer()
default_dir = '.'
timestep = 0
Lx, Ly = (60, 60)
source = ColumnDataSource(data={})

def update_files(attr, old, new):
    fname.options = new
    if new:
        fname.value = new[0]
    update_trajectory(None, None, fname.value)

def update_trajectory(attr, old, new):
    global trj
    trj = gsd.hoomd.open(
        str(Path(directory.value) / new), 'rb')
    index.end = len(trj) - 1
    if index.value > len(trj) - 1:
        update_index(None, None, len(trj)-1)
    else:
        update_index(None, None, index.value)

def update_index(attr, old, new):
    update_snapshot(attr, old, int(new))

def update_snapshot(attr, old, new):
    if old != new:
        global snapshot
        snapshot = trj[new]
        update_data(None, None, None)

def update_data(attr, old, new):
    p.title.text = f'Timestep: {snapshot.configuration.step:.3g}'

    source.data = snapshot2data(snapshot,
                                molecule=molecule,
                                extra_particles=extra_particles)


def update_directory(attr, old, new):
    files = sorted([filename.name for filename in Path(new).glob('dump*.gsd')])
    update_files(None, None, files)


directory = TextInput( value=default_dir, title='Source directory', width=300,)
directory.on_change('value', update_directory)

fname = Select(title='File', value='', options=[])
fname.on_change('value', update_trajectory)

index = Slider(title='Index', value=0, start=0, end=1, step=1)
index.on_change('value', update_index)

radius_scale = Slider(title='Particle Radius', value=1, start=0.1, end=2, step=0.05)
radius_scale.on_change('value', update_data)


# When using webgl as the backend the save option doesn't work for some reason.
p = figure(x_range=(-Ly/2, Ly/2), y_range=(-Ly/2, Ly/2),
           active_scroll='wheel_zoom', width=800, height=800,
           title=f'Timestep: {timestep:.2g}')

update_directory(None, None, default_dir)

plot_circles(p, source)

controls = widgetbox([directory, fname, index, radius_scale], width=300)
layout = row(controls, p)

curdoc().add_root(layout)
curdoc().title = "Crossfilter"
