#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Test the simulation module
"""

from statdyn import Simulation, initialise
import pytest


def run_npt_test():
    run_npt(initialise.init_from_none().take_snapshot(),
            3.00,
            1000,
            )



