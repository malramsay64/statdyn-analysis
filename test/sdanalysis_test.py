#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the sdrun command line tools."""

import subprocess


def test_comp_dynamics():
    command = ['sdanalysis',
               'comp_dynamics',
               '-v',
               '-o', 'test/output',
               'test/data/trajectory-13.50-3.00.gsd',
               ]
    ret = subprocess.run(command)
    assert ret.returncode == 0
