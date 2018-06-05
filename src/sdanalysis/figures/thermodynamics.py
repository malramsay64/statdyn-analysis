#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
#
# pylint: skip-file
"""Bokeh dashboard for interactive visualisation of thermodynamic properties."""

import logging
from pathlib import Path

import numpy as np
import pandas
from bokeh.layouts import column, gridplot, row, widgetbox
from bokeh.models import (
    CheckboxButtonGroup,
    ColumnDataSource,
    MultiSelect,
    Panel,
    Select,
    Tabs,
    TextInput,
)
from bokeh.plotting import curdoc, figure

logger = logging.getLogger(__name__)


def read_file(source):
    """Read a file into a pandas dataframe."""
    data = pandas.read_table(source, sep="\t")
    data["time"] = pandas.to_timedelta(data.timestep * 0.005, unit="us")
    data.set_index("time", drop=True, inplace=True)
    data = data[~data.index.duplicated(keep="last")]
    data = data.resample("1ms").mean()
    return data


def update_file_list(attr, old, new):
    old_file = fname.value
    fname.options = [filename.name for filename in Path(new).glob("thermo*")]
    try:
        new_file = fname.options[0]
        if old_file in fname.options:
            new_file = old_file
        fname.value = new_file
    except IndexError:
        pass


def update_file(attr, old, new):
    global dataframe
    src_fname = Path(directory.value) / new
    logger.debug("Read file %s", src_fname)
    dataframe = read_file(src_fname)
    update_factors(None, None, None)
    update_datacolumns(None, None, None)


def update_factors(attr, old, new):
    factors.options = list(dataframe.columns)
    factors.value = factors.options[2]


def update_datacolumns(attr, old, new):
    default_columns.data = {
        "x": dataframe.timestep.values,
        "temperature": dataframe.temperature.values,
        "pressure": dataframe.pressure.values,
        "potential_energy": dataframe.potential_energy.values,
        "kinetic_energy": dataframe.kinetic_energy.values,
    }
    datacolumns.data = {
        "x": dataframe.timestep.values,
        "y": dataframe[factors.value].values,
    }


dataframe = pandas.DataFrame()
default_columns = ColumnDataSource(data={})
datacolumns = ColumnDataSource(data={})
directory = TextInput(value=".", title="Source Directory")
directory.on_change("value", update_file_list)
fname = Select(title="File", value="", options=[])
fname.on_change("value", update_file)
factors = Select(options=[], value="")
factors.on_change("value", update_datacolumns)
update_file_list(None, None, directory.value)
logger.debug("Defualt colums %s", default_columns.data)
logger.debug("Defualt colummns columns %s", list(default_columns.data.keys()))
cols = list(default_columns.data.keys())
try:
    cols.remove("x")
except ValueError:
    pass
x_range = None
fig_args = {
    "plot_height": 150,
    "plot_width": 800,
    "tools": "pan, box_zoom, xwheel_zoom",
    "active_scroll": "xwheel_zoom",
}
default_plots = [figure(**fig_args, y_axis_label=col) for col in cols]
for fig, col in zip(default_plots, cols):
    fig.line(source=default_columns, x="x", y=col)
    if x_range is not None:
        fig.x_range = x_range
    x_range = fig.x_range
logger.debug("Laying out defualt tab")
controls_default = widgetbox([directory, fname], width=300)
grid = gridplot(default_plots, ncols=1)
default_layout = row(controls_default, grid)
default_tab = Panel(child=default_layout, title="Default")
logger.debug("Creating investigate_tab")
fig_args["plot_height"] = 400
plot = figure(**fig_args)
plot.line(source=datacolumns, x="x", y="y")
controls_inv = widgetbox([factors], width=300)
inv_layout = column(controls_inv, plot)
investigate_tab = Panel(child=inv_layout, title="Investigate")
tabs = Tabs(tabs=[default_tab, investigate_tab])
curdoc().add_root(tabs)
curdoc().title = "Thermodynamics"
