#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Module for setting up and running a hoomd simulation."""

import logging

import hoomd
import numpy as np

from . import initialise
from ..StepSize import GenerateStepSeries
from .helper import (SimulationParams, dump_frame, set_dump, set_integrator,
                     set_thermo)

logger = logging.getLogger(__name__)


def run_npt(snapshot: hoomd.data.SnapshotParticleData,
            context: hoomd.context.SimulationContext,
            sim_params: SimulationParams,
            ) -> None:
    """Initialise and run a hoomd npt simulation.

    Args:
        snapshot (class:`hoomd.data.snapshot`): Hoomd snapshot object
        context:
        sim_params:


    """
    with context:
        initialise.initialise_snapshot(
            snapshot=snapshot,
            context=context,
            molecule=sim_params.molecule,
        )
        set_integrator(sim_params)
        set_thermo(sim_params.filename(prefix='thermo'),
                   thermo_period=sim_params.output_interval)
        set_dump(sim_params.filename(prefix='dump'),
                 dump_period=sim_params.output_interval,
                 group=sim_params.group)
        if sim_params.dynamics:
            iterator = GenerateStepSeries(sim_params.num_steps,
                                          num_linear=100,
                                          max_gen=sim_params.max_gen,
                                          gen_steps=20000,
                                          )
            # Zeroth step
            curr_step = iterator.next()
            assert curr_step == 0
            dumpfile = dump_frame(
                sim_params.filename(prefix='trajectory'),
                group=sim_params.group
            )
            for curr_step in iterator:
                hoomd.run_upto(curr_step, quiet=True)
                dumpfile.write_restart()
        else:
            hoomd.run(sim_params.num_steps)
        dump_frame(sim_params.filename(), group=sim_params.group)


def read_snapshot(context: hoomd.context.SimulationContext,
                  fname: str,
                  rand: bool=False
                  ) -> hoomd.data.SnapshotParticleData:
    """Read a hoomd snapshot from a hoomd gsd file.

    Args:
    fname (string): Filename of GSD file to read in
    rand (bool): Whether to randomise the momenta of all the particles

    Returns:
    class:`hoomd.data.SnapshotParticleData`: Hoomd snapshot

    """
    with context:
        snapshot = hoomd.data.gsd_snapshot(fname)
        if rand:
            nbodies = snapshot.particles.body.max() + 1
            np.random.shuffle(snapshot.particles.velocity[:nbodies])
            np.random.shuffle(snapshot.particles.angmom[:nbodies])
            return snapshot
