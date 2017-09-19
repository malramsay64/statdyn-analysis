#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the molecule class."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats

from statdyn import molecules


@pytest.fixture(params=[molecules.Molecule, molecules.Trimer, molecules.Dimer])
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


@given(floats(min_value=0, allow_infinity=False, allow_nan=False))
def test_moment_inertia_scaling(scaling_factor):
    """Test that the scaling factor is working properly."""
    reference = molecules.Trimer()
    scaled = molecules.Trimer(moment_inertia_scale=scaling_factor)
    assert len(reference.moment_inertia) == len(scaled.moment_inertia)
    with np.errstate(over='ignore'):
        assert np.allclose(np.array(reference.moment_inertia)*scaling_factor,
                           np.array(scaled.moment_inertia))
