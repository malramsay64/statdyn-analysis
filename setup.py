#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Command line tool to run simulations."""

from setuptools import setup, find_packages


setup(
    name='statdyn',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    include_package_data=True,
    entry_points="""
        [console_scripts]
        sdrun=statdyn.sdrun.main:main
    """,
)
