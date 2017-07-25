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
               'equilibrium',
               'test/data/Trimer-13.50-3.00.gsd',
               '-v',
               '-t', '3.00',
               '--no-dynamics',
               '-s', '100',
               '-o', 'test/output',
               ]
    ret = subprocess.run(command)
    assert ret.returncode == 0


def test_create():
    """Ensure sdrun liquid works."""
    command = ['sdrun',
               'create',
               '-v',
               '-t', '2.50',
               '-s', '100',
               'test/output/test_create.gsd',
               ]
    ret = subprocess.run(command)
    assert ret.returncode == 0
    ret = subprocess.run(command + ['--interface'])
    assert ret.returncode == 0


def test_interface():
    """Ensure sdrun liquid works."""
    subprocess.run(['ls', 'test/data'])
    command = ['sdrun',
               'interface',
               'test/data/Trimer-13.50-3.00.gsd',
               '-v',
               '-t', '1.50',
               '--no-dynamics',
               '-s', '100',
               '-o', 'test/output',
               ]
    ret = subprocess.run(command)
    assert ret.returncode == 0
