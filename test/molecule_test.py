#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the molecule class."""

import pytest
from statdyn import molecule


@pytest.fixture(params=[molecule.Molecule, molecule.Trimer, molecule.Dimer])
def mol_setup(request):
    """Test molecule setup."""
    return request.param()


def test_mol_types(mol_setup):  # pylint: disable=redefined-outer-name
    """Test mol_types."""
    assert isinstance(mol_setup.get_types(), list)


def test_moment_inertia(mol_setup):  # pylint: disable=redefined-outer-name
    """Test moment_inertia."""
    assert isinstance(mol_setup.moment_inertia, tuple)
    assert len(mol_setup.moment_inertia) == 3
