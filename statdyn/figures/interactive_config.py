#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
#
# pylint: skip-file

"""Create an interactive view of a configuration."""

import logging
from functools import partial
from pathlib import Path

import gsd.hoomd
from bokeh.layouts import column, row, widgetbox
from bokeh.models import Button, ColumnDataSource, Select, Slider, Toggle
from bokeh.plotting import curdoc, figure
from tornado import gen

from statdyn.figures.configuration import plot, plot_circles, snapshot2data
from statdyn.molecules import Trimer

logger = logging.getLogger(__name__)

# Definition of initial state
trj = None
snapshot = None
extra_particles = True
molecule = Trimer()
default_dir = '.'
timestep = 0
Lx, Ly = (60, 60)
source = ColumnDataSource(data={'x': [0], 'y': [0], 'radius': [1], 'colour': ['red']})
play = False
doc = curdoc()


def get_filename():
    return str(Path.cwd() / directory.value / fname.value)


def update_files(attr, old, new):
    fname.options = new
    if new:
        fname.value = new[0]
    update_trajectory(None, None, fname.value)


def update_trajectory(attr, old, new):
    global trj
    trj = gsd.hoomd.open(get_filename(), 'rb')
    # Bokeh will cope with IndexError in file but not beginning and end
    # of slider being the same value.
    index.end = max(len(trj) - 1, 1)
    if index.value > len(trj) - 1:
        update_index(None, None, len(trj)-1)
    else:
        update_index(None, None, index.value)


def update_index(attr, old, new):
    update_snapshot(attr, old, int(new))


def incr_index():
    if index.value < index.end:
        index.value += increment_size.value


def decr_index():
    if index.value > index.start:
        index.value -= increment_size.value


def update_snapshot(attr, old, new):
    if old != new:
        global snapshot
        try:
            snapshot = trj[new]
        except IndexError:
            pass
        update_data(None, None, None)


@gen.coroutine
def update_source(data):
    source.data = data


def update_data(attr, old, new):
    p.title.text = 'Timestep: {:.5g}'.format(snapshot.configuration.step)

    data = snapshot2data(snapshot,
                         molecule=molecule,
                         extra_particles=extra_particles,
                         ordering=ordered.active,
                         )
    source.data = data


def update_data_now(arg):
    update_data(None, None, None)


def update_directory(attr, old, new):
    files = sorted([filename.name for filename in Path(directory.value).glob('dump*.gsd')])
    if files:
        update_files(None, None, files)


def play_pause_toggle(arg):
    if arg:
        doc.add_periodic_callback(incr_index, 100)
    else:
        doc.remove_periodic_callback(incr_index)


DIR_OPTIONS = sorted([d.parts[-1] for d in Path.cwd().glob('*/') if d.is_dir() and len(list(d.glob('dump*.gsd')))])
directory = Select(value=DIR_OPTIONS[-1], title='Source directory', options=DIR_OPTIONS)
directory.on_change('value', update_directory)

fname = Select(title='File', value='', options=[])
fname.on_change('value', update_trajectory)

index = Slider(title='Index', value=0, start=0, end=1, step=1)
index.on_change('value', update_index)

ordered = Toggle(name='order', label='Colour Order')
ordered.on_click(update_data_now)

radius_scale = Slider(title='Particle Radius', value=1, start=0.1, end=2, step=0.05)
radius_scale.on_change('value', update_data)

play_pause = Toggle(name='Play/Pause', label="Play/Pause")
play_pause.on_click(play_pause_toggle)

nextFrame = Button(label='Next')
nextFrame.on_click(incr_index)

prevFrame = Button(label='Previous')
prevFrame.on_click(decr_index)

increment_size = Slider(title='Increment Size', value=1, start=1, end=100, step=1)

media = widgetbox([prevFrame, play_pause, nextFrame, increment_size], width=300)


# When using webgl as the backend the save option doesn't work for some reason.
p = figure(width=920, height=800, aspect_scale=1, match_aspect=True,
           title='Timestep: {:.2g}'.format(timestep),
           output_backend='webgl',
           active_scroll='wheel_zoom')

update_directory(None, None, default_dir)
update_data(None, None, None)

plot_circles(p, source)

controls = widgetbox([directory, fname, index, ordered], width=300)
layout = row(column(controls, media), p)

doc.add_root(layout)
doc.title = "Configurations"
