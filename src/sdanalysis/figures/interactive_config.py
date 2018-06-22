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

import functools
import logging
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

import gsd.hoomd
import numpy as np
from bokeh.layouts import column, row, widgetbox
from bokeh.models import (
    Button,
    ColorBar,
    ColumnDataSource,
    Div,
    FixedTicker,
    LinearColorMapper,
    RadioButtonGroup,
    Select,
    Slider,
    Toggle,
)
from bokeh.plotting import curdoc, figure
from hsluv import hpluv_to_hex
from tornado import gen

from ..frame import HoomdFrame
from ..molecules import Trimer
from ..order import (
    compute_ml_order,
    compute_voronoi_neighs,
    dt_model,
    knn_model,
    orientational_order,
)
from ..util import get_filename_vars, variables
from .configuration import DARK_COLOURS, frame2data, plot_circles, plot_frame

logger = logging.getLogger(__name__)
gsdlogger = logging.getLogger("gsd")
gsdlogger.setLevel("WARN")


def parse_directory(directory: Path, glob: str = "*.gsd") -> Dict[str, List]:
    all_values: Dict[str, np.ndarray] = {}
    files = directory.glob(glob)
    for fname in files:
        file_vars = get_filename_vars(fname)
        # Convert named tuples to a dict of lists
        for var_name in file_vars._fields:
            curr_list = all_values.get(var_name)
            if curr_list is None:
                curr_list = []
            curr_list.append(getattr(file_vars, var_name))
            all_values[var_name] = curr_list
    for key, value in all_values.items():
        logger.debug("Key: %s has value: %s", key, value)
        all_values[key] = sorted(list(set(value)))
    return all_values


def variables_to_file(file_vars: variables, directory: Path) -> Path:
    if file_vars.crystal is not None:
        glob_pattern = (
            f"dump-Trimer-P{file_vars.pressure}-T{file_vars.temperature}-{file_vars.crystal}.gsd"
        )
    else:
        glob_pattern = f"-P{file_vars.pressure}-T{file_vars.temperature}"
    return next(directory.glob(glob_pattern))


def compute_neigh_ordering(box, positions, orientations):
    return compute_voronoi_neighs(box, positions) == 6


class TrimerFigure(object):
    order_functions = {
        "None": None,
        "Orient": functools.partial(orientational_order, order_threshold=0.75),
        "Decision Tree": functools.partial(compute_ml_order, dt_model()),
        "KNN Model": functools.partial(compute_ml_order, knn_model()),
        "Num Neighs": compute_neigh_ordering,
    }
    controls_width = 400

    def __init__(self, doc, directory: Path = None) -> None:

        self._doc = doc
        self.plot = None
        self._trajectory = [None]

        if directory is None:
            directory = Path.cwd()
        self._source = ColumnDataSource(
            {"x": [], "y": [], "orientation": [], "colour": []}
        )
        self.directory = directory
        self.initialise_directory()

        self.initialise_trajectory_interface()
        self.update_current_trajectory(None, None, None)
        self._playing = False

        # Initialise user interface
        self.initialise_media_interface()

        self.initialise_doc()

    def initialise_directory(self) -> None:
        self.variable_selection = parse_directory(self.directory, glob="*.gsd")
        logger.debug("Variables present: %s", self.variable_selection.keys())

        self._pressure_button = RadioButtonGroup(
            name="Pressure",
            labels=self.variable_selection["pressure"],
            active=0,
            width=self.controls_width,
        )
        self._temperature_button = RadioButtonGroup(
            name="Temperature",
            labels=self.variable_selection["temperature"],
            active=0,
            width=self.controls_width,
        )
        self._crystal_button = RadioButtonGroup(
            name="Crystal",
            labels=self.variable_selection["crystal"],
            active=0,
            width=self.controls_width,
        )

        self._pressure_button.on_change("active", self.update_current_trajectory)
        self._temperature_button.on_change("active", self.update_current_trajectory)
        self._crystal_button.on_change("active", self.update_current_trajectory)

    def create_files_interface(self) -> None:
        directory_name = Div(
            text=f"<b>Current Directory:</b><br/>{self.directory.stem}",
            width=self.controls_width,
        )
        file_selection = column(
            directory_name,
            Div(text="<b>Pressure:</b>"),
            self._pressure_button,
            Div(text="<b>Temperature:</b>"),
            self._temperature_button,
            Div(text="<b>Crystal Structure:</b>"),
            self._crystal_button,
            Div(text="<hr/>", width=self.controls_width, height=10),
            height=400,
        )
        return file_selection

    def get_selected_variables(self) -> variables:
        return variables(
            temperature=self.variable_selection["temperature"][
                self._temperature_button.active
            ],
            pressure=self.variable_selection["pressure"][self._pressure_button.active],
            crystal=self.variable_selection["crystal"][self._crystal_button.active],
        )

    def get_selected_file(self) -> Path:
        return variables_to_file(self.get_selected_variables(), self.directory)

    def update_frame(self, attr, old, new) -> None:
        self._frame = HoomdFrame(self._trajectory[self.index])
        self.update_data(None, None, None)

    def radio_update_frame(self, attr) -> None:
        self.update_frame(attr, None, None)

    @property
    def index(self) -> int:
        try:
            return self._trajectory_slider.value
        except AttributeError:
            return 0

    def initialise_trajectory_interface(self) -> None:
        self._order_parameter = RadioButtonGroup(
            name="Classification algorithm:",
            labels=list(self.order_functions.keys()),
            active=0,
            width=self.controls_width,
        )
        self._order_parameter.on_click(self.radio_update_frame)

    def create_trajectory_interface(self) -> None:
        return column(
            Div(text="<b>Classification Algorithm:<b>"),
            self._order_parameter,
            Div(text="<hr/>", width=self.controls_width, height=10),
            height=120,
        )

    def update_current_trajectory(self, attr, old, new) -> None:
        self._trajectory = gsd.hoomd.open(str(self.get_selected_file()), "rb")
        self.update_frame(attr, old, new)

    def initialise_media_interface(self) -> None:
        self._trajectory_slider = Slider(
            title="Trajectory Index",
            value=0,
            start=0,
            end=max(len(self._trajectory), 1),
            step=1,
            width=self.controls_width,
        )
        self._trajectory_slider.on_change("value", self.update_frame)

        self._play_pause = Toggle(
            name="Play/Pause", label="Play/Pause", width=int(self.controls_width / 3)
        )
        self._play_pause.on_click(self._play_pause_toggle)
        self._nextFrame = Button(label="Next", width=int(self.controls_width / 3))
        self._nextFrame.on_click(self._incr_index)
        self._prevFrame = Button(label="Previous", width=int(self.controls_width / 3))
        self._prevFrame.on_click(self._decr_index)
        self._increment_size = Slider(
            title="Increment Size",
            value=10,
            start=1,
            end=100,
            step=1,
            width=self.controls_width,
        )

    def _incr_index(self) -> None:
        if self._trajectory_slider.value < self._trajectory_slider.end:
            self._trajectory_slider.value = min(
                self._trajectory_slider.value + self._increment_size.value,
                self._trajectory_slider.end,
            )

    def _decr_index(self) -> None:
        if self._trajectory_slider.value > self._trajectory_slider.start:
            self._trajectory_slider.value = max(
                self._trajectory_slider.value - self._increment_size.value,
                self._trajectory_slider.start,
            )

    def create_media_interface(self):
        #  return widgetbox([prevFrame, play_pause, nextFrame, increment_size], width=300)
        return column(
            Div(text="<b>Media Controls:</b>"),
            self._trajectory_slider,
            row(
                [self._prevFrame, self._play_pause, self._nextFrame],
                width=int(self.controls_width),
            ),
            self._increment_size,
        )
        # When using webgl as the backend the save option doesn't work for some reason.

    def _update_source(self, data):
        logger.debug("Data Keys: %s", data.keys())
        self._source.data = data

    def get_order_function(self) -> Optional[Callable]:
        return self.order_functions[
            list(self.order_functions.keys())[self._order_parameter.active]
        ]

    def update_data(self, attr, old, new):
        if self.plot:
            self.plot.title.text = f"Timestep {self._frame.timestep:.5g}"
        data = frame2data(
            self._frame, order_function=self.get_order_function(), molecule=Trimer()
        )
        self._update_source(data)

    def update_data_attr(self, attr):
        self.update_data(attr, None, None)

    def _play_pause_toggle(self, attr):
        if self._playing:
            self._doc.remove_periodic_callback(self._incr_index)
            self._playing = False
        else:
            self._doc.add_periodic_callback(self._incr_index, 100)
            self._playing = True

    def create_legend(self):
        cm_orient = LinearColorMapper(palette=DARK_COLOURS, low=-np.pi, high=np.pi)
        cm_class = LinearColorMapper(
            palette=[hpluv_to_hex((0, 0, 60)), hpluv_to_hex((0, 0, 80))], low=0, high=2
        )

        plot = figure(width=200, height=250)
        plot.toolbar_location = None
        plot.border_fill_color = "#FFFFFF"
        plot.outline_line_alpha = 0
        cb_orient = ColorBar(
            title="Orientation",
            major_label_text_font_size="10pt",
            title_text_font_style="bold",
            color_mapper=cm_orient,
            orientation="horizontal",
            ticker=FixedTicker(ticks=[-np.pi, 0, np.pi]),
            major_label_overrides={-np.pi: "-π", 0: "0", np.pi: "π"},
            width=100,
            major_tick_line_color=None,
            location=(0, 120),
        )
        cb_class = ColorBar(
            color_mapper=cm_class,
            title="Classification",
            major_label_text_font_size="10pt",
            title_text_font_style="bold",
            orientation="vertical",
            ticker=FixedTicker(ticks=[0.5, 1.5]),
            major_label_overrides={0.5: "Crystal", 1.5: "Liquid"},
            label_standoff=15,
            major_tick_line_color=None,
            width=20,
            height=80,
            location=(0, 0),
        )
        plot.add_layout(cb_orient)
        plot.add_layout(cb_class)
        return plot

    def initialise_doc(self):
        self.plot = figure(
            width=920,
            height=800,
            aspect_scale=1,
            match_aspect=True,
            title=f"Timestep {self._frame.timestep:.5g}",
            output_backend="webgl",
            active_scroll="wheel_zoom",
        )
        self.plot.xgrid.grid_line_color = None
        self.plot.ygrid.grid_line_color = None
        self.plot.x_range.start = -30
        self.plot.x_range.end = 30
        self.plot.y_range.start = -30
        self.plot.y_range.end = 30
        plot_circles(self.plot, self._source)

    def create_doc(self):
        self.update_data(None, None, None)
        controls = column(
            [
                self.create_files_interface(),
                self.create_trajectory_interface(),
                self.create_media_interface(),
            ],
            width=int(self.controls_width * 1.1),
        )
        self._doc.add_root(
            row(controls, self.plot, self.create_legend(), create_reference_interface())
        )
        self._doc.title = "Configurations"


def create_reference_interface():
    div_str = "<p><b>Reference Structures:</b></p>"
    div_width = 200
    for crystal in ["p2", "p2gg", "pg"]:
        crys_img = f"https://malramsay.com/static/img/molecules/crystal-{crystal}.png"
        div_str += f"""
            <figure style="text-align:center" width={div_width}>
                <img src="{crys_img}" width={div_width} height={div_width}>
                <figcaption align="center">{crystal}</figcaption>
            </figure>
            """
    return Div(text=div_str, width=div_width + 100)


def make_document(doc):
    fig = TrimerFigure(doc)
    fig.create_doc()
