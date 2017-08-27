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

from statdyn.figures.colour import clean_orientation, colour_orientation

logger = logging.getLogger(__name__)


DEFAULT_DIR = '.'


def update_trj(attr, old, new):
    global trj
    trj = gsd.hoomd.open(
        str(Path(directory.value) / fname.value), 'rb')
    index.end = len(trj) - 1
    if index.value > len(trj) - 1:
        index.value = len(trj) - 1
    update_data(attr, old, new)


def update_data(attr, old, new):
    snap = trj[int(index.value)]
    p.title.text = f'Timestep: {snap.configuration.step:.3g}'
    data = {
        'x': snap.particles.position[:, 0],
        'y': snap.particles.position[:, 1],
        'radius': ((snap.particles.typeid * -0.362444) + 1)*radius_scale.value,
    }
    try:
        data['orientation'] = colour_orientation(clean_orientation(snap))
    except AttributeError:
        data['orientation'] = data['radius']

    source.data = data


def update_files(attr, old, new):
    global files
    files = sorted([filename.name for filename in Path(directory.value).glob('dump*.gsd')])
    fname.options = files
    if files:
        fname.value = files[0]


def update_all():
    curr_file = None
    if fname:
        curr_file = fname.value
    update_files(None, None, None)
    if curr_file in files:
        fname.value = curr_file
    update_trj(None, None, None)
    update_data(None, None, None)


directory = TextInput(
    value=DEFAULT_DIR,
    title='Source directory',
    width=300,
)
directory.on_change('value', update_files)

fname = Select(title='File', value='', options=[])
fname.on_change('value', update_trj)

index = Slider(title='Index', value=0, start=0, end=1, step=1)
index.on_change('value', update_data)

radius_scale = Slider(title='Particle Radius', value=1, start=0.1, end=2, step=0.05)
radius_scale.on_change('value', update_data)

refresh = Button(label='Refresh')
refresh.on_click(update_all)

timestep = 0
Lx, Ly = (60, 60)

source = ColumnDataSource(data={})

# When using webgl as the backend the save option doesn't work for some reason.
p = figure(x_range=(-Ly/2, Ly/2), y_range=(-Ly/2, Ly/2),
           active_scroll='wheel_zoom', width=800, height=800,
           title=f'Timestep: {timestep:.2g}')

update_all()

p.circle('x', 'y', radius='radius',
         fill_alpha=1, fill_color='orientation',
         line_color=None, source=source)

controls = widgetbox([directory, fname, index, radius_scale, refresh], width=300)
layout = row(controls, p)

curdoc().add_root(layout)
curdoc().title = "Crossfilter"
