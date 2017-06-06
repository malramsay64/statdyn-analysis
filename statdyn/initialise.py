#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Module for initialisation of a hoomd simulation environment

This module allows the initialisation from a number of different starting
configurations, whether that is a file, a crystal lattice, or no predefined
config.
"""
from statdyn import molecule, crystals
import numpy as np
import hoomd

def set_defaults(kwargs):
    kwargs.setdefault('mol', molecule.Trimer())
    kwargs.setdefault('cell_len', 4)
    kwargs.setdefault('cell_dimensions', (10, 10))
    kwargs.setdefault('timesep', 0)
    kwargs.setdefault('cmd_args', '')


def init_from_file(fname, **kwargs):
    set_defaults(kwargs)
    context = kwargs.get('context', hoomd.context.initialize(kwargs.get('cmd_args')))
    snapshot = hoomd.data.gsd_snapshot(fname, kwargs.get('timestep', 0))
    sys = init_from_snapshot(snapshot, **kwargs)
    return sys


def init_from_none(**kwargs):
    set_defaults(kwargs)
    context = kwargs.get('context', hoomd.context.initialize(kwargs.get('cmd_args')))
    with context:
        sys = hoomd.init.create_lattice(
            unitcell=hoomd.lattice.sq(a=kwargs.get('cell_len')),
            n=kwargs.get('cell_dimensions')
        )
        snap = sys.take_snapshot(all=True)
    return init_from_snapshot(snap, **kwargs)


def init_from_snapshot(snapshot, **kwargs):
    """Initialise the configuration from a snapshot

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
    mol = kwargs.get('mol')
    create_bodies = False
    if num_particles == num_mols:
        create_bodies = True
    else:
        assert num_particles % mol.num_particles == 0
    snapshot = _check_properties(snapshot, mol)
    sys = hoomd.init.read_snapshot(snapshot)
    mol.define_potential()
    mol.define_dimensions()
    rigid = mol.define_rigid()
    rigid.create_bodies(create=create_bodies)
    return sys

def init_from_crystal(crystal, **kwargs):
    kwargs.setdefault('mol', crystal.molecule)
    set_defaults(kwargs)
    context = kwargs.get('context', hoomd.context.initialize(kwargs.get('cmd_args')))
    with context:
        sys = hoomd.init.create_lattice(
            unitcell=crystal.get_unitcell(),
            n=kwargs.get('cell_dimensions')
        )
        snap = sys.take_snapshot(all=True)
    return init_from_snapshot(snap, **kwargs)


def _check_properties(snapshot, mol):
    num_atoms = snapshot.particles.N
    snapshot.particles.types = mol.get_types()
    snapshot.particles.moment_inertia[:num_atoms+1] = np.array(
        [mol.moment_inertia]*num_atoms)
    return snapshot
