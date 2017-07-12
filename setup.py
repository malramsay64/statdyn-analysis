#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Command line tool to run simulations."""

from setuptools import setup


setup(
    name='statdyn',
    version='0.1',
    py_modules=['statdyn'],
    install_requires=[
        'Click',
    ],
    entry_points="""
        [console_scripts]
        sdrun=statdyn.sdrun.main:main
    """,
)
