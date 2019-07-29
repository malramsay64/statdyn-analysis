#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Command line tool to run simulations."""

import os

from setuptools import find_packages, setup


def get_version():
    g = {}
    exec(open("src/sdanalysis/version.py").read(), g)
    return g["__version__"]


def read(path, encoding="utf-8"):
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path, encoding=encoding) as fp:
        return fp.read()


install_requires = [
    "numpy~=1.16",
    "scipy>=1.0",
    "scikit-learn>=0.20.0",
    "pandas",
    "tables>=3.5.1",
    "bokeh>=1.0",
    "gsd>=1.3.0",
    "pyyaml>=5.1",
    "hsluv",
    "attrs",
    "click~=7.0.0",
    "freud-analysis~=1.0.0",
    "rowan",
    "tqdm",
    "joblib~=0.13.2",
]
test_requires = [
    "pytest",
    "pylint",
    "hypothesis",
    "coverage",
    "black==19.3b0",
    "mypy",
    "pytest-mypy",
    "pytest-pylint",
    "pytest-cov",
    "pytest-lazy-fixture",
]
docs_requires = ["sphinx", "sphinx_rtd_theme", "sphinx_autodoc_typehints"]
dev_requires = docs_requires + test_requires

setup(
    name="sdanalysis",
    version=get_version(),
    python_requires=">=3.6",
    install_requires=install_requires,
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    extras_require={"docs": docs_requires, "tests": test_requires, "dev": dev_requires},
    entry_points="""
        [console_scripts]
        sdanalysis=sdanalysis.main:sdanalysis
    """,
    url="https://github.com/malramsay64/statdyn-analysis",
    author="Malcolm Ramsay",
    author_email="malramsay64@gmail.com",
    description="Statistical dynamics analysis of molecular dynamics trajectories.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
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
