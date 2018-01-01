Statdyn Analysis
================

[![Build Status](https://travis-ci.org/malramsay64/statdyn-analysis.svg?branch=master)](https://travis-ci.org/malramsay64/statdyn-analysis)
[![codecov](https://codecov.io/gh/malramsay64/statdyn-analysis/branch/master/graph/badge.svg)](https://codecov.io/gh/malramsay64/statdyn-analysis)
[![Anaconda-Server Badge](https://anaconda.org/malramsay/sdanalysis/badges/installer/conda.svg)](https://conda.anaconda.org/malramsay)
[![Anaconda-Server Badge](https://anaconda.org/malramsay/sdanalysis/badges/version.svg)](https://anaconda.org/malramsay/sdanalysis)


This is a set of scripts that use
[Hoomd](https://bitbucket.org/glotzer/hoomd-blue) to perform the Molecular
dynamics simulations of a glass forming molecular liquid. There is a particular
focus on understanding the dynamic properties of these molecules.

Note that this is still very early alpha software and there are likely to be
large breaking changes that occur.

Installation
------------

The simplest method of installation is using `conda`. To install

    conda install -c malramsay statdyn-analysis

It is also possible to set the repository up as a development environment,
in which case cloning the repository and installing is possible by running

    git clone https://github.com/malramsay64/statdyn-analysis.git
    cd statdyn-analysis
    conda env create
    source activate sdanalysis-dev
    python setup.py develop

Once the environment is setup the tests can be run with

    pytest

Running Analysis
-------------------

Dynamics of a trajectory can be computed using the command

    sdanalysis comp-dynamics trajectory-Trimer-13.50-1.20.gsd

which will generate an hdf5 file of the same name containing a single table,
`dynamics` which has all the dynamic quantities tabulated. This also includes
a start index, over which statistics can be computed.

Finally the command

    sdanalysis figure

will open up a bokeh server which will allow for the interactive visualisation
of all `dump-*.gsd` files in the current directory.

