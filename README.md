MD-Molecules-Hoomd
==================

This is a set of scripts that use [Hoomd](https://bitbucket.org/glotzer/hoomd-blue) to perform the Molecular dynamics simulations of a glass forming molecular liquid. There is a particular focus on understanding the dynamic properties of these molecules.

Installation
------------

There are some prerequisites for this set of scripts to run properly
 - hoomd-blue version 1.3.3
 - R
 - python numpy
 - python scipy
and to build the documentation
 - python sphinx
 - python sphinx-rtd-theme

Running Simulations
-------------------

To set up the equilibration runs

    $ mkdir scratch
    $ cd scratch
    $ cp mol.xml .
    $ hoomd ../TrimerEquil.hoomd

Note that using the default set of temperatures and steps will result in an equilibration that will take on the order of 3-5 days depending on configuration. There is a similar timescale for the production runs as well.

To set up the production runs, which require the files generated in the equilibration runs, the command

    $ hoomd ../TrimerDynamics.hoomd

will run the simulation.

The figures resulting from the dynamics can be produced with

    $ Rscript ../figures/dynamics.R

which creates a file `dynamics.pdf` in the current directory.

Configuring Simulations
-----------------------

To configure the temperatures and timesteps of the equilibration, edit the `STEPS` and `TEMPERATURES` variables in either the `TrimerDynamics.hoomd` or the `TrimerEquil.hoomd` files.

The `STEPS` variables is a scaling factor for all simulations. It is a base value that applies to each temperature. The `TEMPERATURES` variable contains a list of *tuples*, which contain the temperature and a multiplier\* the `STEPS` variable. This configuration takes into account the slowing down of the dynamics at low temperatures, needing increasingly longer simulation times to reach equilibration.
