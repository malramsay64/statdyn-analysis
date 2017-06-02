#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Module to test the molecule class
"""

from statdyn import molecule
import pytest

@pytest.fixture(params=[molecule.Molecule, molecule.Trimer, molecule.Dimer])
def mol_setup(request):
    return request.param()

def test_mol_types(mol_setup):
    assert isinstance(mol_setup.get_types(), list)

def test_moment_inertia(mol_setup):
    assert isinstance(mol_setup.moment_inertia, tuple)
    assert len(mol_setup.moment_inertia) == 3

