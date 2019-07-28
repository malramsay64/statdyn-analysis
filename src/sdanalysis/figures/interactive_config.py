#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
#
# pylint: disable=unused-argument, too-many-public-methods, too-many-instance-attributes
"""Create an interactive view of a configuration.

This is a visualisation of a trajectory, allowing interaction with both the current
frame, zooming in at particular points, and additionally exploring the time dimension of
a trajectory.

This is intended to be a simple and easy to use interface, where the visualisation is
nice and clear, and interaction is intuitive. To this end, while there are tool-tips available,
providing the information on the position, orientation and further details of particles,
they are off by default.

While the particles are trimer in shape, for reasons of performance they are rendered as
3 circles. The performance gain is from using the WebGL for the rendering of the
circles, while rendering of arbitrary shapes is not supported.

"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gsd.hoomd
import numpy as np
from bokeh.layouts import column, row
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
from bokeh.plotting import figure
from hsluv import hpluv_to_hex

from ..frame import HoomdFrame
from ..molecules import Trimer
from ..order import create_ml_ordering, create_neigh_ordering, create_orient_ordering
from ..util import parse_directory
from .configuration import DARK_COLOURS, frame2data, plot_circles

logger = logging.getLogger(__name__)
gsdlogger = logging.getLogger("gsd")
gsdlogger.setLevel("WARN")


class TrimerFigure:
    order_functions: Dict[str, Any] = {
        "None": None,
        "Orient": create_orient_ordering(threshold=0.75),
        "Num Neighs": create_neigh_ordering(neighbours=6),
    }
    controls_width = 400

    _frame = None
    plot = None
    _temperatures = None
    _pressures = None
    _crystals = None
    _iter_index = None

    _callback = None

    def __init__(self, doc, directory: Path = None, models=None) -> None:
        self._doc = doc
        self._trajectory = [None]

        if directory is None:
            directory = Path.cwd()
        self._source = ColumnDataSource(
            {"x": [], "y": [], "orientation": [], "colour": [], "radius": []}
        )

        if models is not None:
            if not isinstance(models, (list, tuple)):
                raise ValueError("The argument models has to have type list or tuple")

            logger.debug("Found additional models: %s", models)
            for model in models:
                model = Path(model)
                self.order_functions[model.stem] = create_ml_ordering(model)

        self.directory = directory
        self.initialise_directory()
        self._filename_div = Div(text="", width=self.controls_width)

        self.initialise_trajectory_interface()
        self.update_current_trajectory(None, None, None)
        self._playing = False

        # Initialise user interface
        self.initialise_media_interface()

        self.initialise_doc()

    def initialise_directory(self) -> None:
        self.variable_selection = parse_directory(self.directory, glob="dump*.gsd")
        logger.debug("Pressures present: %s", self.variable_selection.keys())

        self._pressures = sorted(list(self.variable_selection.keys()))
        self._pressure_button = RadioButtonGroup(
            name="Pressure ",
            labels=self._pressures,
            active=0,
            width=self.controls_width,
        )
        self._pressure_button.on_change("active", self.update_temperature_button)
        pressure = self._pressures[self._pressure_button.active]

        self._temperatures = sorted(list(self.variable_selection[pressure].keys()))
        self._temperature_button = Select(
            name="Temperature",
            options=self._temperatures,
            value=self._temperatures[0],
            width=self.controls_width,
        )
        self._temperature_button.on_change("value", self.update_crystal_button)

        temperature = self._temperature_button.value
        self._crystals = sorted(
            list(self.variable_selection[pressure][temperature].keys())
        )
        self._crystal_button = RadioButtonGroup(
            name="Crystal", labels=self._crystals, active=0, width=self.controls_width
        )
        self._crystal_button.on_change("active", self.update_index_button)

        crystal = self._crystals[self._crystal_button.active]
        self._iter_index = sorted(
            list(self.variable_selection[pressure][temperature][crystal].keys())
        )
        self._iter_index_button = Select(
            name="Iteration Index",
            options=self._iter_index,
            value=self._iter_index[0],
            width=self.controls_width,
        )
        self._iter_index_button.on_change("value", self.update_current_trajectory)

    @property
    def pressure(self) -> Optional[str]:
        if self._pressures is None:
            return None
        return self._pressures[self._pressure_button.active]

    @property
    def temperature(self) -> Optional[str]:
        if self._temperatures is None:
            return None
        return self._temperature_button.value

    @property
    def crystal(self) -> Optional[str]:
        logger.debug(
            "Current crystal %s from %s", self._crystal_button.active, self._crystals
        )
        if self._crystals is None:
            return None
        return self._crystals[self._crystal_button.active]

    @property
    def iter_index(self) -> Optional[str]:
        logger.debug(
            "Current index %s from %s", self._iter_index_button.value, self._iter_index
        )
        return self._iter_index_button.value

    def update_temperature_button(self, attr, old, new):
        self._temperatures = sorted(list(self.variable_selection[self.pressure].keys()))

        self._temperature_button.options = self._temperatures
        self._temperature_button.value = self._temperatures[0]
        self.update_crystal_button(None, None, None)

    def update_crystal_button(self, attr, old, new):
        self._crystals = sorted(
            list(self.variable_selection[self.pressure][self.temperature].keys())
        )

        self._crystal_button.labels = self._crystals
        self._crystal_button.active = 0
        self.update_index_button(None, None, None)

    def update_index_button(self, attr, old, new):
        self._iter_index = sorted(
            list(
                self.variable_selection[self.pressure][self.temperature][
                    self.crystal
                ].keys()
            )
        )
        self._iter_index_button.options = self._iter_index
        self._iter_index_button.value = self._iter_index[0]
        self.update_current_trajectory(None, None, None)

    def create_files_interface(self) -> None:
        directory_name = Div(
            text=f"<b>Current Directory:</b><br/>{self.directory}",
            width=self.controls_width,
        )
        self._filename_div = Div(text="", width=self.controls_width)
        current_file = self.get_selected_file()
        if current_file is not None:
            self._filename_div.text = f"<b>Current File:</b><br/>{current_file.name}"
            file_selection = column(
                directory_name,
                self._filename_div,
                Div(text="<b>Pressure:</b>"),
                self._pressure_button,
                Div(text="<b>Temperature:</b>"),
                self._temperature_button,
                Div(text="<b>Crystal Structure:</b>"),
                self._crystal_button,
                Div(text="<b>Iteration Index:</b>"),
                self._iter_index_button,
            )
        return file_selection

    def get_selected_file(self) -> Optional[Path]:
        if self.pressure is None:
            return None
        if self.temperature is None:
            return None
        return self.variable_selection[self.pressure][self.temperature][self.crystal][
            self.iter_index
        ]

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
        logger.debug("Loading Models: %s", self.order_functions.keys())
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
        if self.get_selected_file() is not None:
            logger.debug("Opening %s", self.get_selected_file())
            self._trajectory = gsd.hoomd.open(str(self.get_selected_file()), "rb")
            num_frames = len(self._trajectory)

            try:
                if self._trajectory_slider.value > num_frames:
                    self._trajectory_slider.value = num_frames - 1
                self._trajectory_slider.end = len(self._trajectory) - 1
            except AttributeError:
                pass

            self.update_frame(attr, old, new)
            current_file = self.get_selected_file()
            if current_file is not None:
                self._filename_div.text = (
                    f"<b>Current File:</b><br/>{current_file.name}"
                )
            else:
                self._filename_div.text = f"<b>Current File:</b><br/>None"

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
        if self.plot and self._frame is not None:
            self.plot.title.text = f"Timestep {self._frame.timestep:,}"
        if self._frame is not None:
            data = frame2data(
                self._frame, order_function=self.get_order_function(), molecule=Trimer()
            )
            self._update_source(data)

    def update_data_attr(self, attr):
        self.update_data(attr, None, None)

    def _play_pause_toggle(self, attr):
        if self._playing:
            self._doc.remove_periodic_callback(self._callback)
            self._playing = False
        else:
            self._callback = self._doc.add_periodic_callback(self._incr_index, 100)
            self._playing = True

    @staticmethod
    def create_legend():
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
            title=f"Timestep {0:.5g}",
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
        self._doc.add_root(row(controls, self.plot, self.create_legend()))
        self._doc.title = "Configurations"


def make_document(doc, directory: Path = None, models: Optional[List[Path]] = None):
    if models is None:
        models = []
    fig = TrimerFigure(doc, directory=directory, models=models)
    fig.create_doc()
