#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test interactive bokeh configurations run."""

import subprocess


def test_interactive_null():
    import sdanalysis.figures.interactive_config  # pylint: disable=unused-variable


def test_thermodynamics_null():
    import sdanalysis.figures.thermodynamics  # pylint: disable=unused-variable


def test_interactive():
    subprocess.check_call(['python', '-c', 'import sdanalysis.figures.interactive_config'],
                          cwd='test')


def test_thermodynamics():
    subprocess.check_call(['python', '-c', 'import sdanalysis.figures.thermodynamics'],
                          cwd='test/data')
