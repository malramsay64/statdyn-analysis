#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""A collection of utility functions."""

from pathlib import Path
from typing import NamedTuple, Optional


class variables(NamedTuple):
    temperature: Optional[str]
    pressure: Optional[str]
    crystal: Optional[str]


def get_filename_vars(fname: Path):
    fname = Path(fname)
    flist = fname.stem.split("-")
    if len(flist) < 4:
        return variables(None, None, None)

    pressure_str = flist[2]
    if pressure_str[0] == "P":
        pressure: Optional[str] = pressure_str[1:]
    else:
        pressure = None

    temperature_str = flist[3]
    if temperature_str[0] == "T":
        temp: Optional[str] = temperature_str[1:]
    else:
        temp = None

    if len(flist) >= 5:
        crys: Optional[str] = flist[4]
    else:
        crys = None

    return variables(temp, pressure, crys)
