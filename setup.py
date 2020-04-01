#! /usr/bin/env python3
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
    "numpy>=1.14,<1.19",
    "scipy~=1.0",
    "scikit-learn>=0.20.0,<1.0",
    "pandas>=0.23,<0.26",
    "tables>=3.5.1",
    "bokeh>=1.0",
    "gsd>=1.3",
    "pyyaml>=5.1",
    "hsluv==0.0.2",
    "attrs>=19.2,<19.4",
    "click~=7.0.0",
    "freud-analysis>=1.0,<1.3",
    "rowan>=1.2,<1.3",
    "tqdm>=4,<5",
    "joblib>=0.13.2,<0.15",
]
test_requires = [
    "pytest>=5.0.1,<5.5.0",
    "pylint>=2.4.3,<2.5",
    "hypothesis>=4.43.1,<6.0",
    "coverage==5.0.1",
    "isort==4.3.21",
    "black==19.10b0",
    "mypy==0.770",
    "pytest-cov==2.8.1",
    "flake8==3.7.9",
    "flake8-docstrings==1.5.0",
    "pydocstyle!=4.0.0,<6.0.0",  # There is an issue with indented code blocks in 4.0.0
]
docs_requires = [
    "sphinx>=2.1.2,<2.4.0",
    "sphinx_rtd_theme==0.4.3",
    "sphinx_autodoc_typehints==1.10.3",
]
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
        "Programming Language :: Python :: 3.7",
    ],
)
