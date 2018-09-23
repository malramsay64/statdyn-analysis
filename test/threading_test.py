#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

from pathlib import Path

from sdanalysis.threading import _set_input_file


def test_set_input_file(sim_params):
    infile = Path("test.in")
    result_params = _set_input_file(sim_params, infile)
    assert result_params.infile == infile
    assert sim_params.infile != infile
    init_temp = result_params.temperature
    result_params.temperature = 0.1
    assert sim_params.temperature == init_temp
    assert result_params.temperature == 0.1
