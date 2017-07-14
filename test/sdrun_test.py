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
    command = ['sdrun', 'crystal', '-v', '-s 100', '-t 1.50', '--no-dynamics']
    ret = subprocess.run(command)
    assert ret.returncode == 0


def test_liquid():
    """Ensure sdrun liquid works."""
    subprocess.run(['ls', 'test/data'])
    command = ['sdrun', 'liquid', '-v', '-c', 'test/data',
               '-t', '3.00', '--no-dynamics', '-s', '100']
    ret = subprocess.run(command)
    assert ret.returncode == 0
