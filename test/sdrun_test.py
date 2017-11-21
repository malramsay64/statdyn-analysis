#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the sdrun command line tools."""

import subprocess
import sys

import pytest

from statdyn.sdrun.main import parse_args

TEST_ARGS = [
    ['prod',
     'test/data/Trimer-13.50-3.00.gsd',
     '-t', '3.00',
     '--no-dynamics',
     ],
    ['create',
     '-t', '2.50',
     '--space-group', 'pg',
     '--lattice-lengths', '20', '24',
     'test/output/test_create.gsd',
     ],
    ['equil',
     '-t', '2.50',
     'test/data/Trimer-13.50-3.00.gsd',
     'test/output/test_equil.gsd',
     ]
]

COMMON_ARGS = [
     '--hoomd-args', '"--mode=cpu"',
     '-o', 'test/output',
     '-s', '100',
     '-v',
]


@pytest.mark.parametrize('arguments', TEST_ARGS)
def test_man(arguments):
    func, sim_params = parse_args(arguments + COMMON_ARGS)
    func(sim_params)


@pytest.mark.parametrize('arguments', TEST_ARGS)
def test_commands(arguments):
    """Ensure sdrun prod works."""
    subprocess.run(['ls', 'test/data'])
    command = ['sdrun'] + arguments + COMMON_ARGS
    ret = subprocess.run(command)
    assert ret.returncode == 0


@pytest.mark.skipif(sys.platform == 'darwin', reason='No MPI support on macOS')
@pytest.mark.parametrize('arguments', TEST_ARGS)
def test_commands_mpi(arguments):
    """Ensure sdrun prod works."""
    subprocess.run(['ls', 'test/data'])
    command = ['mpirun', '-np', '4', 'sdrun'] + arguments + COMMON_ARGS
    ret = subprocess.run(command)
    assert ret.returncode == 0


def test_comp_dynamics():
    command = ['sdrun',
               'comp_dynamics',
               '-v',
               '-o', 'test/output',
               '--hoomd-args', '"--mode=cpu"',
               'test/data/trajectory-13.50-3.00.gsd',
               ]
    ret = subprocess.run(command)
    assert ret.returncode == 0


def test_sdrun_figure():
    command = ['sdrun', 'figure']
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(command, timeout=1)
