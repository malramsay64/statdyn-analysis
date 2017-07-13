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

import hoomd
import hoomd.md as md
import numpy as np

from .. import molecule


logger = logging.getLogger(__name__)


def set_defaults(kwargs):
    """Set the argument defaults."""
    kwargs.setdefault('mol', molecule.Trimer())
    kwargs.setdefault('cell_len', 4)
    kwargs.setdefault('cell_dimensions', (10, 10))
    kwargs.setdefault('timesep', 0)
    kwargs.setdefault('init_args', '')


def init_from_file(fname, **kwargs):
    """Initialise a hoomd simulation from an input file."""
    set_defaults(kwargs)
    context = kwargs.get('context',
                         hoomd.context.initialize(kwargs.get('init_args')))
    snapshot = hoomd.data.gsd_snapshot(str(fname), kwargs.get('timestep', 0))
    sys = init_from_snapshot(snapshot, context=context, **kwargs)
    return sys


def init_from_none(**kwargs):
    """Initialise a system from no inputs.

    This creates a simulation with a large unit cell lattice such that there
    is no chance of molecules overlapping and places molecules on the lattice.
    """
    set_defaults(kwargs)
    context = kwargs.get(
        'context', hoomd.context.initialize(kwargs.get('init_args')))
    with context:
        sys = hoomd.init.create_lattice(
            unitcell=hoomd.lattice.sq(a=kwargs.get('cell_len')),
            n=kwargs.get('cell_dimensions')
        )
        snap = sys.take_snapshot(all=True)
    return init_from_snapshot(snap, **kwargs)


def init_from_snapshot(snapshot, **kwargs):
    """Initialise the configuration from a snapshot.

    In this function it is checked that the data in the snapshot and the
    passed arguments are in agreement with each other, and rectified if not.
    """
    set_defaults(kwargs)
    if not kwargs.get('context'):
        hoomd.context.initialize('')
    num_particles = snapshot.particles.N
    try:
        num_mols = max(snapshot.particles.bodies)
    except AttributeError:
        num_mols = num_particles
    logger.debug(f'Number of molecules: {num_mols}')
    mol = kwargs.get('mol')
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
    rigid.create_bodies(create=create_bodies)
    if create_bodies:
        logger.info('Rigid bodies created')
    return sys


def init_from_crystal(crystal, **kwargs):
    """Initialise a hoomd simulation from a crystal lattice.

    Args:
        crystal (class:`statdyn.crystals.Crystal`): The crystal lattice to
            generate the simulation from.
    """
    kwargs.setdefault('mol', crystal.molecule)
    set_defaults(kwargs)
    context1 = hoomd.context.initialize(kwargs.get('init_args'))
    with context1:
        logger.debug(f"Creating {crystal} cell of size {kwargs.get('cell_dimensions')}")
        sys = hoomd.init.create_lattice(
            unitcell=crystal.get_unitcell(),
            n=kwargs.get('cell_dimensions')
        )
        snap = sys.take_snapshot(all=True)
        sys = init_from_snapshot(snap, **kwargs)
        md.integrate.mode_minimize_fire(hoomd.group.rigid_center(), dt=0.005)
        hoomd.run(1000)
        equil_snap = sys.take_snapshot(all=True)
    context2 = kwargs.get('context',
                          hoomd.context.initialize(kwargs.get('init_args')))
    with context2:
        snap = make_orthorhombic(equil_snap)
        sys = init_from_snapshot(snap, **kwargs)
    return sys


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
    len_x = snapshot.box.Lx
    len_y = snapshot.box.Ly
    len_z = snapshot.box.Lz
    xlen = len_x + snapshot.box.xy * len_y
    pos = snapshot.particles.position
    pos += np.array([xlen / 2., len_y / 2., len_z / 2.])
    pos = pos % np.array([len_x, len_y, len_z])
    pos -= np.array([len_x / 2., len_y / 2., len_z / 2.])
    snapshot.particles.position[:] = pos
    box = hoomd.data.boxdim(len_x, len_y, len_z, 0, 0, 0, dimensions=2)
    hoomd.data.set_snapshot_box(snapshot, box)
    return snapshot


def _check_properties(snapshot, mol):
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
