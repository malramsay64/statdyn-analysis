# -*- coding: utf-8 -*-

from __future__ import absolute_import

from bokeh.core.enums import Anchor, Direction, StepMode
from bokeh.core.has_props import abstract
from bokeh.core.properties import (
    AngleSpec,
    Bool,
    DistanceSpec,
    Enum,
    Float,
    Include,
    Instance,
    Int,
    NumberSpec,
    Override,
    String,
    StringSpec,
)
from bokeh.core.property_mixins import (
    FillProps,
    LineProps,
    ScalarFillProps,
    ScalarLineProps,
    TextProps,
)
from bokeh.model import Model
from bokeh.models.glyphs import XYGlyph
from bokeh.models.mappers import ColorMapper, LinearColorMapper

# XXX: allow `from bokeh.models.glyphs import *`
from bokeh.models.markers import (
    Asterisk,
    Circle,
    CircleCross,
    CircleX,
    Cross,
    Diamond,
    DiamondCross,
    Hex,
    InvertedTriangle,
    Marker,
    Square,
    SquareCross,
    SquareX,
    Triangle,
    X,
)


class Trimer(XYGlyph):
    u""" Render ellipses.

    """

    __implementation__ = "trimer.ts"

    # a canonical order for positional args that can be used for any
    # functions derived from this class
    _args = ("x", "y", "width", "height", "angle")

    x = NumberSpec(
        help="""
    The x-coordinates of the centers of the ellipses.
    """
    )

    y = NumberSpec(
        help="""
    The y-coordinates of the centers of the ellipses.
    """
    )

    width = DistanceSpec(
        help="""
    The widths of each ellipse.
    """
    )

    height = DistanceSpec(
        help="""
    The heights of each ellipse.
    """
    )

    angle = AngleSpec(
        default=0.0,
        help="""
    The angle the ellipses are rotated from horizontal. [rad]
    """,
    )

    line_props = Include(
        LineProps,
        use_prefix=False,
        help="""
    The %s values for the ovals.
    """,
    )

    fill_props = Include(
        FillProps,
        use_prefix=False,
        help="""
    The %s values for the ovals.
    """,
    )
