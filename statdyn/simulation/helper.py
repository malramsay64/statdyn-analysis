#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Series of helper functions for the initialisation of parameters."""

import logging
from pathlib import Path

import hoomd
import hoomd.md as md

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.WARNING)


def set_integrator(temperature: float,
                   tau: float=1.,
                   pressure: float=13.50,
                   tauP: float=1.,
                   step_size: float=0.005,
                   prime_interval: int=33533,
                   group: hoomd.group.group=None,
                   crystal: bool=False,
                   ) -> hoomd.md.integrate.npt:
    """Hoomd integrate method."""
    if group is None:
        group = hoomd.group.rigid_center()
    md.update.enforce2d()
    if prime_interval:
        md.update.zero_momentum(period=prime_interval, phase=-1)
    md.integrate.mode_standard(step_size)
    integrator = md.integrate.npt(
        group=group,
        kT=temperature,
        tau=tau,
        P=pressure,
        tauP=tauP,
    )
    if crystal:
        integrator.set_params(
            rescale_all=True,
            couple='none',
        )
    return integrator


def dump_frame(outfile: Path,
               timestep: int=0,
               group: hoomd.group.group=None,
               ) -> None:
    """Dump frame to file."""
    if group is None:
        group = hoomd.group.rigid_center()
    hoomd.dump.gsd(
        str(outfile),
        period=None,
        time_step=timestep,
        group=group,
        static=['topology', 'attribute']
    )
