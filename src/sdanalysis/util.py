#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""A collection of utility functions."""

from collections import namedtuple
from pathlib import Path
from typing import NamedTuple


class variables(NamedTuple):
    temperature: float
    pressure: float
    crystal: str


def get_filename_vars(fname: Path):
    fname = Path(fname)
    flist = fname.stem.split("-")
    if len(flist) < 3:
        return variables(None, None, None)

    pressure = flist[2][1:]
    temp = flist[3][1:]
    try:
        crys = flist[4]
    except IndexError:
        crys = None
    return variables(temp, pressure, crys)
