#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the sdrun command line tools."""

import subprocess

import pytest


def test_comp_dynamics():
    command = ['sdanalysis',
               'comp_dynamics',
               '-v',
               '-o', 'test/output',
               'test/data/trajectory-13.50-3.00.gsd',
               ]
    ret = subprocess.run(command)
    assert ret.returncode == 0


def test_sdrun_figure():
    command = ['sdanalysis', 'figure']
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(command, timeout=1)
