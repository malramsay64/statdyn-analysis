#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

import pytest

from sdanalysis import molecules

MOLECULE_LIST = [
    molecules.Molecule,
    molecules.Trimer,
    molecules.Dimer,
    molecules.Disc,
    molecules.Sphere,
]


@pytest.fixture(scope="module", params=MOLECULE_LIST)
def mol(request):
    return request.param()
