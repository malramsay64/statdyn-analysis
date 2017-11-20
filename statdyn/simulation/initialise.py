#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Module for initialisation of a hoomd simulation environment.

This module allows the initialisation from a number of different starting
configurations, whether that is a file, a crystal lattice, or no predefined
config.
"""
import logging
from pathlib import Path
from typing import Tuple

import hoomd
import hoomd.md as md
import numpy as np

from .. import molecules
from .helper import SimulationParams, dump_frame

logger = logging.getLogger(__name__)


def init_from_file(fname: Path,
                   hoomd_args: str='',
                   ) -> hoomd.data.SnapshotParticleData:
    """Initialise a hoomd simulation from an input file."""
    logger.debug('Initialising from file %s', fname)
    # Hoomd context needs to be initialised before calling gsd_snapshot
    logger.debug('Hoomd Arguments: %s', hoomd_args)
    temp_context = hoomd.context.initialize(hoomd_args)
    with temp_context:
        return hoomd.data.gsd_snapshot(str(fname), frame=0)


def init_from_none(hoomd_args: str='',
                   cell_len: float=4,
                   cell_dimensions: Tuple[int, int]=(20, 30),
                   ) -> hoomd.data.SnapshotParticleData:
    """Initialise a system from no inputs.

    This creates a simulation with a large unit cell lattice such that there
    is no chance of molecules overlapping and places molecules on the lattice.
    """
    logger.debug('Hoomd Arguments: %s', hoomd_args)
    with hoomd.context.initialize(hoomd_args):
        sys = hoomd.init.create_lattice(
            unitcell=hoomd.lattice.sq(a=cell_len),
            n=cell_dimensions
        )
        return sys.take_snapshot(all=True)


def initialise_snapshot(snapshot: hoomd.data.SnapshotParticleData,
                        context: hoomd.context.SimulationContext,
                        molecule: molecules.Molecule,
                        ) -> hoomd.data.system_data:
    """Initialise the configuration from a snapshot.

    In this function it is checked that the data in the snapshot and the
    passed arguments are in agreement with each other, and rectified if not.
    """
    with context:
        try:
            num_particles = snapshot.particles.N
            num_mols = max(snapshot.particles.bodies)
        except AttributeError:
            num_particles = len(snapshot.particles.position)
            num_mols = num_particles
        logger.debug('Number of particles: %d , Number of molecules: %d', num_particles, num_mols)
        create_bodies = False
        if num_particles == num_mols:
            logger.info('Creating rigid bodies')
            create_bodies = True
        else:
            assert num_particles % molecule.num_particles == 0
        snapshot = _check_properties(snapshot, molecule)
        sys = hoomd.init.read_snapshot(snapshot)
        molecule.define_potential()
        molecule.define_dimensions()
        rigid = molecule.define_rigid()
        if rigid:
            rigid.create_bodies(create=create_bodies)
            if create_bodies:
                logger.info('Rigid bodies created')
        return sys


def init_from_crystal(sim_params: SimulationParams,
                      ) -> hoomd.data.SnapshotParticleData:
    """Initialise a hoomd simulation from a crystal lattice.

    Args:
        crystal (class:`statdyn.crystals.Crystal`): The crystal lattice to
            generate the simulation from.
    """
    logger.info('Hoomd Arguments: %s', sim_params.hoomd_args)
    assert hasattr(sim_params, 'cell_dimensions')
    assert hasattr(sim_params, 'crystal')
    assert hasattr(sim_params, 'molecule')
    temp_context = hoomd.context.initialize(sim_params.hoomd_args)
    with temp_context:
        logger.debug("Creating %s cell of size %s",
                     sim_params.crystal, sim_params.cell_dimensions)

        sys = hoomd.init.create_lattice(
            unitcell=sim_params.crystal.get_unitcell(),
            n=sim_params.cell_dimensions
        )
        snap = sys.take_snapshot(all=True)
    temp_context = hoomd.context.initialize(sim_params.hoomd_args)
    with temp_context:
        sys = initialise_snapshot(snap, temp_context, sim_params.molecule)
        md.integrate.mode_standard(dt=sim_params.step_size)

        md.integrate.npt(group=sim_params.group,
                         kT=sim_params.temperature,
                         xy=True,
                         couple='none',
                         P=sim_params.pressure,
                         tau=sim_params.tau,
                         tauP=sim_params.tauP,
                         )

        equil_snap = sys.take_snapshot(all=True)
        dump_frame(sim_params.filename(), group=sim_params.group)
    return make_orthorhombic(equil_snap)


def make_orthorhombic(snapshot: hoomd.data.SnapshotParticleData
                      ) -> hoomd.data.SnapshotParticleData:
    """Create orthorhombic unit cell from snapshot.

    This uses the periodic boundary conditions of the cell to generate an
    orthorhombic simulation cell from the input simulation environment. This
    is to ensure consistency within simulations and because it is simpler to
    use an orthorhombic simulation cell in calculations.

    Todo:
        This function doesn't yet account for particles within a molecule
        which are accross a simulation boundary. This needs to be fixed before
        this function is truly general, otherwise it only works with special
        cells.

    """
    logger.debug('Snapshot type: %s', snapshot)
    len_x = snapshot.box.Lx
    len_y = snapshot.box.Ly
    len_z = snapshot.box.Lz
    xlen = len_x + snapshot.box.xy * len_y
    snapshot.particles.position[:, 0] += xlen/2.
    snapshot.particles.position[:, 0] %= len_x
    snapshot.particles.position[:, 0] -= len_x/2.

    logger.debug('Updated positions: \n%s', snapshot.particles.position)
    box = hoomd.data.boxdim(len_x, len_y, len_z, 0, 0, 0, dimensions=2)
    hoomd.data.set_snapshot_box(snapshot, box)
    return snapshot


def _check_properties(snapshot: hoomd.data.SnapshotParticleData,
                      molecule: molecules.Molecule
                      ) -> hoomd.data.SnapshotParticleData:
    try:
        nbodies = len(snapshot.particles.body)
        logger.debug('number of rigid bodies: %d', nbodies)
        snapshot.particles.types = molecule.get_types()
        snapshot.particles.moment_inertia[:nbodies] = np.array(
            [molecule.moment_inertia] * nbodies)
    except (AttributeError, ValueError):
        num_atoms = len(snapshot.particles.position)
        logger.debug('num_atoms: %d', num_atoms)
        if num_atoms > 0:
            snapshot.particles.types = molecule.get_types()
            snapshot.particles.moment_inertia[:] = np.array(
                [molecule.moment_inertia] * num_atoms)
    return snapshot
