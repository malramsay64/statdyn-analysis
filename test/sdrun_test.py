#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the sdrun command line tools."""


import subprocess


def test_crystal():
    """Ensure command line tools work."""
    command = ['sdrun',
               'crystal',
               '-v',
               '-s', '100',
               '-t', '1.50',
               '--no-dynamics',
               '-o', 'test/output',
               ]
    ret = subprocess.run(command)
    assert ret.returncode == 0


def test_liquid():
    """Ensure sdrun liquid works."""
    subprocess.run(['ls', 'test/data'])
    command = ['sdrun',
               'liquid',
               'test/data/Trimer-13.50-3.00.gsd',
               '-v',
               '-t', '3.00',
               '--no-dynamics',
               '-s', '100',
               '-o', 'test/output',
               ]
    ret = subprocess.run(command)
    assert ret.returncode == 0
