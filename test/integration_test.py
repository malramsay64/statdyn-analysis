#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Testing that everything works together."""

from pathlib import Path

import pytest

from sdanalysis.main import comp_dynamics, comp_relaxations


@pytest.fixture()
def trajectory():
    return Path(__file__).parent / "data/trajectory-Trimer-P13.50-T3.00.gsd"


def test_dynamics(runner, trajectory):
    result = runner.invoke(comp_dynamics, [str(trajectory)])
    assert result.exit_code == 0


def test_relaxation(runner, trajectory):
    result = runner.invoke(comp_dynamics, [str(trajectory)])
    assert result.exit_code == 0
    result = runner.invoke(comp_relaxations, ["dynamics.h5"])
    assert result.exit_code == 0, result.output
