MD-Molecules-Hoomd
==================

[![Build Status](https://travis-ci.org/malramsay64/MD-Molecules-Hoomd.svg?branch=master)](https://travis-ci.org/malramsay64/MD-Molecules-Hoomd)
[![codecov](https://codecov.io/gh/malramsay64/MD-Molecules-Hoomd/branch/master/graph/badge.svg)](https://codecov.io/gh/malramsay64/MD-Molecules-Hoomd)
[![Anaconda-Server Badge](https://anaconda.org/malramsay/statdyn/badges/installer/conda.svg)](https://conda.anaconda.org/malramsay)
[![Anaconda-Server Badge](https://anaconda.org/malramsay/statdyn/badges/version.svg)](https://anaconda.org/malramsay/statdyn)


This is a set of scripts that use
[Hoomd](https://bitbucket.org/glotzer/hoomd-blue) to perform the Molecular
dynamics simulations of a glass forming molecular liquid. There is a particular
focus on understanding the dynamic properties of these molecules.

Note that this is still very early alpha software and there are likely to be
large breaking changes that occur.

Installation
------------

The simplest method of installation is using `conda`. To install

    conda install -c malramsay statdyn

It is also possible to set the repository up as a development environment,
in which case cloning the repository and installing is possible by running

    git clone https://github.com/malramsay64/MD-Molecules-Hoomd.git
    cd MD-Molecules-Hoomd
    conda env create
    source activate statdyn-dev
    python setup.py develop

Once the environment is setup the tests can be run with

    pytest

Running Simulations
-------------------

Interaction with the program is currently through the command line, using the
command line arguments to specify the various parameters.

To create a crystal structure for a simulation run

    sdrun create --space-group p2 -s 1000 test.gsd

which will generate a file which has a trimer molecule with a p2 crystal
structure. The simulation will be run for 1000 steps at a default low
temperature to relax any stress.

For other options see

    sdrun create --help

This output file we created can then be equilibrated using

    sdrun equil -t 1.2 -s 1000 test.gsd test-1.2.gsd

which will gradually bring the temperature from the default to 1.2 over 1000
steps with the final configuration output to `test-1.2.gsd`. This is unlikely
to actually equilibrate this configuration, but it will run fast.

A production run can be run with the `prod` sub-command

    sdrun prod -t 1.2 -s 1000 test-1.2.gsd

This has a different series of options including outputting a series of
timesteps optimised for the analysis of dynamics quantities in the file
prefixed with `trajectory-`. This dynamics optimised file can be analysed
with

    sdrun comp-dynamics trajectory-Trimer-13.50-1.20.gsd

which will generate an hdf5 file of the same name containing a single table,
`dynamics` which has all the dynamic quantities tabulated. This also includes
a start index, over which statistics can be computed.

Finally the command

    sdrun figure

will open up a bokeh server which will allow for the interactive visualisation
of all `dump-*.gsd` files in the current directory.

