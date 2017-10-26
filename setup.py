#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Command line tool to run simulations."""

import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

extensions = [
    Extension(
        'statdyn.analysis.order',
        ['statdyn/analysis/order.pyx'],
        language='c++',
        libraries=['m', 'voro++'],
        include_dirs=[np.get_include()],
    ),
    Extension(
        'statdyn.math_helper',
         ['statdyn/math_helper.pyx'],
        libraries=['m'],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name='statdyn',
    use_scm_version={'version_scheme': 'post-release'},
    setup_requires=['setuptools_scm', ],
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_package_data=True,
    entry_points="""
        [console_scripts]
        sdrun=statdyn.sdrun.main:sdrun
    """,
)
