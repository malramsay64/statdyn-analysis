#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Command line tool to run simulations."""

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
    setup_requires=[
        'setuptools_scm',
        'cython',
        'numpy',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'pandas',
        'tables',
        'bokeh',
        'matplotlib',
        'gsd',
    ],
    packages=find_packages('src'),
    ext_modules=cythonize(extensions, include_path=['src/']),
    package_dir={'': 'src'},
    include_package_data=True,
    entry_points="""
        [console_scripts]
        sdanalysis=sdanalysis.main:sdanalysis
    """,
    url="https://github.com/malramsay64/statdyn-analysis",
    author="Malcolm Ramsay",
    author_email="malramsay64@gmail.com",
    description="Statistical dynamics analysis of molecular dynamics trajectories.",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
