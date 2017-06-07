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
from statdyn import initialise, molecule, crystals
import pytest
import hoomd
import numpy as np


def create_snapshot():
    return initialise.init_from_none().take_snapshot()


def create_file():
    initialise.init_from_none()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        hoomd.dump.gsd(
            tmp.name,
            period=None,
            overwrite=True,
            group=hoomd.group.all()
        )
        return tmp.name


init_test_params = [
    (initialise.init_from_none, None, {}),
    (initialise.init_from_snapshot, [create_snapshot()], {}),
    (initialise.init_from_file, [create_file()], {}),
    (initialise.init_from_crystal, [crystals.p2()], {'cell_dimensions':(10, 5)}),
]


def test_init_from_none():
    sys = initialise.init_from_none()
    snap = sys.take_snapshot()
    assert isinstance(sys, hoomd.data.system_data)
    assert snap.particles.N == 300

def test_init_from_snapshot():
    sys = initialise.init_from_snapshot(create_snapshot())
    assert isinstance(sys, hoomd.data.system_data)

@pytest.mark.parametrize("init_func, args, kwargs", init_test_params)
def test_init_all(init_func, args, kwargs):
    if args:
        sys = init_func(*args, **kwargs)
    else:
        sys = init_func()
    assert isinstance(sys, hoomd.data.system_data)

@pytest.mark.parametrize("init_func, args, kwargs", init_test_params)
def test_2d(init_func, args, kwargs):
    if args:
        sys = init_func(*args, **kwargs)
    else:
        sys = init_func()
    assert sys.box.dimensions == 2

def test_make_orthorhombic():
    with hoomd.context.initialize():
        snap = create_snapshot()
        assert np.all(initialise._make_orthorhombic(snap).particles.position ==
                    snap.particles.position)
        assert snap.box.xy == 0
        assert snap.box.xz == 0
        assert snap.box.yz == 0
    with hoomd.context.initialize():
        snap_crys = hoomd.init.create_lattice(
            unitcell=crystals.p2().get_unitcell(),
            n=(10, 10)
        ).take_snapshot()
        snap_ortho = initialise._make_orthorhombic(snap_crys)
        assert np.all(snap_ortho.particles.position[:, 0] < snap_ortho.box.Lx/2.)
        assert np.all(snap_ortho.particles.position[:, 0] > -snap_ortho.box.Lx/2.)
        assert np.all(snap_ortho.particles.position[:, 1] < snap_ortho.box.Ly/2.)
        assert np.all(snap_ortho.particles.position[:, 1] > -snap_ortho.box.Ly/2.)
        assert snap.box.xy == 0
        assert snap.box.xz == 0
        assert snap.box.yz == 0



