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

try:
    import numpy as np
    from Cython.Build import cythonize
except ModuleNotFoundError as e:
    print("Numpy and Cython are required to install sdanalysis.")
    raise


def get_version():
    g = {}
    exec(open("src/sdanalysis/version.py").read(), g)
    return g["__version__"]


extensions = [
    Extension(
        "sdanalysis.math_util",
        ["src/sdanalysis/math_util.pyx"],
        libraries=["m"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "sdanalysis._order",
        ["src/sdanalysis/_order.pyx", "src/voro/voro++.cc"],
        language="c++",
        libraries=["m"],
        include_dirs=[np.get_include(), "src/voro"],
    ),
]

test_require = [
    "pytest",
    "pylint",
    "numpy-quaternion",
    "hypothesis",
    "coverage",
    "mypy",
    "pytest-mypy",
    "pytest-pylint",
    "pytest-cov",
    "pytest-lazy-fixture",
]
docs_require = []
dev_require = docs_require + test_require

setup(
    name="sdanalysis",
    version=get_version(),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.13",
        "scipy>=1.0",
        "scikit-learn==0.19.0",
        "pandas",
        "tables",
        "bokeh>=0.12.11",
        "gsd>=1.3.0",
        "pyyaml",
        "hsluv",
        "attrs",
        "click",
    ],
    packages=find_packages("src"),
    ext_modules=cythonize(extensions, include_path=["src/"]),
    package_dir={"": "src"},
    include_package_data=True,
    extras_require={"docs": docs_require, "tests": test_require, "dev": dev_require},
    entry_points="""
        [console_scripts]
        sdanalysis=sdanalysis.main:sdanalysis
    """,
    url="https://github.com/malramsay64/statdyn-analysis",
    author="Malcolm Ramsay",
    author_email="malramsay64@gmail.com",
    description="Statistical dynamics analysis of molecular dynamics trajectories.",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
)
