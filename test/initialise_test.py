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
from statdyn import initialise, molecule
import pytest
import hoomd


def create_snapshot():
    return initialise.init_from_none().take_snapshot()


def create_file():
    initialise.init_from_none()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        hoomd.dump.gsd(tmp.name, period=None, overwrite=True, group=hoomd.group.all())
        return tmp.name

@pytest.fixture(params=[
    (initialise.init_from_none, None),
    (initialise.init_from_snapshot, create_snapshot()),
    (initialise.init_from_file, create_file())
])
def init_all(request):
    return request


def test_init_from_none():
    sys = initialise.init_from_none()
    snap = sys.take_snapshot()
    assert isinstance(sys, hoomd.data.system_data)
    assert snap.particles.N == 300

def test_init_from_snapshot():
    sys = initialise.init_from_snapshot(create_snapshot())
    assert isinstance(sys, hoomd.data.system_data)

@pytest.mark.parametrize("init_func, args", [
    (initialise.init_from_none, None),
    (initialise.init_from_snapshot, [create_snapshot()]),
    (initialise.init_from_file, [create_file()])
])
def test_init_all(init_func, args):
    if args:
        sys = init_func(*args)
    else:
        sys = init_func()
    assert isinstance(sys, hoomd.data.system_data)

@pytest.mark.parametrize("init_func, args", [
    (initialise.init_from_none, None),
    (initialise.init_from_snapshot, [create_snapshot()]),
    (initialise.init_from_file, [create_file()])
])
def test_2d(init_func, args):
    if args:
        sys = init_func(*args)
    else:
        sys = init_func()
    assert sys.box.dimensions == 2

@pytest.mark.parametrize("init_func, args", [
    (initialise.init_from_none, None),
    (initialise.init_from_snapshot, [create_snapshot()]),
    (initialise.init_from_file, [create_file()])
])
def test_kwarg_propogation(init_func, args):
    if args:
        sys = init_func(*args, mol=molecule.Dimer())
    else:
        sys = init_func(mol=molecule.Dimer())
    snap = sys.take_snapshot()
    assert snap.particles.N == 200
