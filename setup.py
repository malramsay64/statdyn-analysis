#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Command line tool to run simulations."""

from setuptools import find_packages, setup

setup(
    name='statdyn',
    setup_requires=[ "setuptools_git >= 0.3", ],
    packages=find_packages(),
    include_package_data=True,
    entry_points="""
        [console_scripts]
        sdrun=statdyn.sdrun.main:sdrun
    """,
)
