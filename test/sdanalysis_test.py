#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Test the sdrun command line tools."""

import subprocess
from tempfile import TemporaryDirectory
import pytest


@pytest.fixture
def output_directory():
    with TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_comp_dynamics(output_directory):
    command = [
        "sdanalysis",
        "comp_dynamics",
        "-v",
        "-o",
        output_directory,
        "test/data/trajectory-Trimer-P13.50-T3.00.gsd",
    ]
    ret = subprocess.run(command)
    assert ret.returncode == 0
