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

from .. import crystals, molecule
from .helper import dump_frame

logger = logging.getLogger(__name__)


def init_from_file(fname: Path,
                   hoomd_args: str='',
                   ) -> hoomd.data.SnapshotParticleData:
    """Initialise a hoomd simulation from an input file."""
    logger.debug(f'Initialising from file {fname}')
    # Hoomd context needs to be initialised before calling gsd_snapshot
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
    with hoomd.context.initialize(hoomd_args):
        sys = hoomd.init.create_lattice(
            unitcell=hoomd.lattice.sq(a=cell_len),
            n=cell_dimensions
        )
        return sys.take_snapshot(all=True)


def initialise_snapshot(snapshot: hoomd.data.SnapshotParticleData,
                        context: hoomd.context.SimulationContext,
                        mol: molecule.Molecule,
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
        logger.debug(f'Number of particles: {num_particles}, Number of molecules: {num_mols}')
        create_bodies = False
        if num_particles == num_mols:
            logger.info('Creating rigid bodies')
            create_bodies = True
        else:
            assert num_particles % mol.num_particles == 0
        snapshot = _check_properties(snapshot, mol)
        sys = hoomd.init.read_snapshot(snapshot)
        mol.define_potential()
        mol.define_dimensions()
        rigid = mol.define_rigid()
        if rigid:
            rigid.create_bodies(create=create_bodies)
            if create_bodies:
                logger.info('Rigid bodies created')
        return sys


def init_from_crystal(crystal: crystals.Crystal,
                      hoomd_args: str='',
                      cell_dimensions: Tuple[int, int]=(30, 34),
                      step_size: float=0.005,
                      optimise_steps: int=1000,
                      outfile: Path=None,
                      ) -> hoomd.data.SnapshotParticleData:
    """Initialise a hoomd simulation from a crystal lattice.

    Args:
        crystal (class:`statdyn.crystals.Crystal`): The crystal lattice to
            generate the simulation from.
    """
    temp_context = hoomd.context.initialize(hoomd_args)
    with temp_context:
        logger.debug(
            f"Creating {crystal} cell of size {cell_dimensions}"
        )

        sys = hoomd.init.create_lattice(
            unitcell=crystal.get_unitcell(),
            n=cell_dimensions
        )
        snap = sys.take_snapshot(all=True)
    temp_context = hoomd.context.initialize(hoomd_args)
    with temp_context:
        sys = initialise_snapshot(snap, temp_context, crystal.molecule)
        md.integrate.mode_standard(dt=step_size)
        temperature = hoomd.variant.linear_interp([
            (0, 0),
            (optimise_steps, 0.5),
        ])
        if crystal.molecule.num_particles == 1:
            group = hoomd.group.all()
        else:
            group = hoomd.group.rigid_center()
        md.integrate.npt(group=group,
                         kT=temperature,
                         xy=True, couple='none',
                         P=13.5, tau=1, tauP=1,
                         )

        equil_snap = sys.take_snapshot(all=True)
        if outfile:
            dump_frame(outfile, group=group)
    return make_orthorhombic(equil_snap)


def init_slab(crystal: crystals.Crystal,
              equil_temp: float,
              equil_steps: int,
              hoomd_args: str='',
              cell_dimensions: Tuple[int, int]=(30, 40),
              melt_temp: float=2.50,
              melt_steps: int=20000,
              tau: float=1.,
              pressure: float=13.50,
              tauP: float=1.,
              step_size: float=0.005,
              ) -> hoomd.data.SnapshotParticleData:
    """Initialise a crystal slab in a liquid."""
    snapshot = init_from_crystal(
        crystal=crystal,
        hoomd_args=hoomd_args,
        cell_dimensions=cell_dimensions,
        step_size=step_size/5,
        optimise_steps=5000,
    )
    temp_context = hoomd.context.initialize(hoomd_args)
    sys = initialise_snapshot(
        snapshot=snapshot,
        context=temp_context,
        mol=crystal.molecule,
    )
    with temp_context:
        md.update.enforce2d()
        prime_interval = 307
        md.update.zero_momentum(period=prime_interval)
        md.integrate.mode_standard(dt=step_size/5)
        group = hoomd.group.intersection(
            'rigid_stationary',
            hoomd.group.cuboid(name='stationary',
                               xmin=-sys.box.Lx/4,
                               xmax=sys.box.Lx/4),
            hoomd.group.rigid_center()
        )
        thermostat = md.integrate.npt(
            group=group,
            kT=melt_temp,
            tau=tau,
            P=pressure,
            tauP=tauP
        )
        hoomd.run(melt_steps/10)
        md.integrate.mode_standard(dt=step_size/5)
        hoomd.run(melt_steps)
        thermostat.set_params(kT=equil_temp)
        hoomd.run(equil_steps)
        return sys.take_snapshot(all=True)


def get_fname(temp: float, ext: str='gsd') -> str:
    """Construct filename of based on the temperature.

    Args:
        temp (float): The temperature of the simulation

    Returns:
        str: The standard filename for my simulations

    """
    return '{mol}-{press:.2f}-{temp:.2f}.{ext}'.format(
        mol='Trimer',
        press=13.50,
        temp=temp,
        ext=ext
    )


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
    logger.debug(f'Snapshot type: {snapshot}')
    len_x = snapshot.box.Lx
    len_y = snapshot.box.Ly
    len_z = snapshot.box.Lz
    xlen = len_x + snapshot.box.xy * len_y
    snapshot.particles.position[:, 0] += xlen/2.
    snapshot.particles.position[:, 0] %= len_x
    snapshot.particles.position[:, 0] -= len_x/2.

    logger.debug(f"Updated positions: \n{snapshot.particles.position}")
    box = hoomd.data.boxdim(len_x, len_y, len_z, 0, 0, 0, dimensions=2)
    hoomd.data.set_snapshot_box(snapshot, box)
    return snapshot


def _check_properties(snapshot: hoomd.data.SnapshotParticleData,
                      mol: molecule.Molecule
                      ) -> hoomd.data.SnapshotParticleData:
    try:
        nbodies = len(snapshot.particles.body)
        logger.debug(f'number of rigid bodies: {nbodies}')
        snapshot.particles.types = mol.get_types()
        snapshot.particles.moment_inertia[:nbodies] = np.array(
            [mol.moment_inertia] * nbodies)
    except (AttributeError, ValueError):
        num_atoms = len(snapshot.particles.position)
        logger.debug(f'num_atoms: {num_atoms}')
        if num_atoms > 0:
            snapshot.particles.types = mol.get_types()
            snapshot.particles.moment_inertia[:] = np.array(
                [mol.moment_inertia] * num_atoms)
    return snapshot
