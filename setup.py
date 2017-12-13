#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Command line tool to run simulations."""

from pathlib import Path
from sysconfig import get_path

from setuptools import find_packages, setup
from setuptools.extension import Extension

import numpy as np
from Cython.Build import cythonize

extensions = [
    Extension(
        'sdanalysis.math_helper',
         ['src/sdanalysis/math_helper.pyx'],
        libraries=['m'],
        include_dirs=[np.get_include()],
    ),
    Extension(
        'sdanalysis._order',
        ['src/sdanalysis/_order.pyx', 'src/voro++/voro++.cc'],
        language='c++',
        libraries=['m'],
        include_dirs=[np.get_include(), 'src/voro++'],
    ),
]

setup(
    name='sdanalysis',
    use_scm_version={'version_scheme': 'post-release'},
    setup_requires=['setuptools_scm', ],
    packages=find_packages('src'),
    ext_modules=cythonize(extensions, include_path=['src/']),
    package_dir={'': 'src'},
    include_package_data=True,
    entry_points="""
        [console_scripts]
        sdanalysis=sdanalysis.main:sdanalysis
    """,
)
