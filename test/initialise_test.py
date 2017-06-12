#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Module for testing the initialisation
"""

import tempfile

import hoomd
import numpy as np
import pytest
from statdyn import crystals, initialise


def create_snapshot():
    """Function to easily create a snapshot for later use in testing"""
    return initialise.init_from_none().take_snapshot()


def create_file():
    """Ease of use function for creating a file for use in testing"""
    initialise.init_from_none()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        hoomd.dump.gsd(
            tmp.name,
            period=None,
            overwrite=True,
            group=hoomd.group.all()
        )
        return tmp.name


INIT_TEST_PARAMS = [
    (initialise.init_from_none, None, {}),
    (initialise.init_from_snapshot, [create_snapshot()], {}),
    (initialise.init_from_file, [create_file()], {}),
    (initialise.init_from_crystal, [
        crystals.TrimerP2()], {'cell_dimensions': (10, 5)}),
]


def test_init_from_none():
    """Test initialisation from none function returns the correct type and
    number of particles"""
    sys = initialise.init_from_none()
    snap = sys.take_snapshot()
    assert isinstance(sys, hoomd.data.system_data)
    assert snap.particles.N == 300


def test_init_from_snapshot():
    """Test initialisation from a snapshot"""
    sys = initialise.init_from_snapshot(create_snapshot())
    assert isinstance(sys, hoomd.data.system_data)


@pytest.mark.parametrize("init_func, args, kwargs", INIT_TEST_PARAMS)
def test_init_all(init_func, args, kwargs):
    """Test the initialisation of all init functions"""
    if args:
        sys = init_func(*args, **kwargs)
    else:
        sys = init_func()
        assert isinstance(sys, hoomd.data.system_data)


@pytest.mark.parametrize("init_func, args, kwargs", INIT_TEST_PARAMS)
def test_2d(init_func, args, kwargs):
    """Test box is 2d when initialised"""
    if args:
        sys = init_func(*args, **kwargs)
    else:
        sys = init_func()
        assert sys.box.dimensions == 2


def test_make_orthorhombic():
    """Ensure that a conversion to an orthorhombic cell goes smoothly

    This tests a number of modes of operation
        - nothing changes in an already orthorhombic cell
        - no particles are outside the box when moved
        - the box is actually orthorhombic
    """
    with hoomd.context.initialize():
        snap = create_snapshot()
        assert np.all(
            initialise._make_orthorhombic(snap).particles.position ==  # pylint: disable=protected-access
            snap.particles.position)
        assert snap.box.xy == 0
        assert snap.box.xz == 0
        assert snap.box.yz == 0
    with hoomd.context.initialize():
        snap_crys = hoomd.init.create_lattice(
            unitcell=crystals.TrimerP2().get_unitcell(),
            n=(10, 10)
        ).take_snapshot()
        snap_ortho = initialise._make_orthorhombic(  # pylint: disable=protected-access
            snap_crys)
        assert np.all(
            snap_ortho.particles.position[:, 0] < snap_ortho.box.Lx / 2.)
        assert np.all(
            snap_ortho.particles.position[:, 0] > -snap_ortho.box.Lx / 2.)
        assert np.all(
            snap_ortho.particles.position[:, 1] < snap_ortho.box.Ly / 2.)
        assert np.all(
            snap_ortho.particles.position[:, 1] > -snap_ortho.box.Ly / 2.)
        assert snap.box.xy == 0
        assert snap.box.xz == 0
        assert snap.box.yz == 0
