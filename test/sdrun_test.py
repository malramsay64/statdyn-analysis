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


def test_prod():
    """Ensure sdrun prod works."""
    subprocess.run(['ls', 'test/data'])
    command = ['sdrun',
               'prod',
               'test/data/Trimer-13.50-3.00.gsd',
               '-v',
               '-t', '3.00',
               '--no-dynamics',
               '-s', '100',
               '-o', 'test/output',
               '--hoomd-args', '"--mode=cpu"',
               ]
    ret = subprocess.run(command)
    assert ret.returncode == 0


@pytest.mark.skipif(sys.platform == 'darwin', reason='No MPI support on macOS')
def test_prod_mpi():
    """Ensure sdrun prod works."""
    subprocess.run(['ls', 'test/data'])
    command = ['sdrun',
               'prod',
               'test/data/Trimer-13.50-3.00.gsd',
               '-v',
               '-t', '3.00',
               '--no-dynamics',
               '-s', '100',
               '-o', 'test/output',
               '--hoomd-args', '"--mode=cpu"',
               ]
    command = 'mpirun -np 4'.split(' ') + command
    ret = subprocess.run(command)
    assert ret.returncode == 0


@pytest.mark.parametrize('space_group', ['p2', 'p2gg', 'pg'])
def test_create(space_group):
    """Ensure sdrun create works."""
    command = ['sdrun',
               'create',
               '-v',
               '-t', '2.50',
               '-s', '100',
               '--space-group', space_group,
               '--lattice-lengths', '20', '24',
               '-o', 'test/output',
               'test/output/test_create.gsd',
               '--hoomd-args', '"--mode=cpu"',
               ]
    ret = subprocess.run(command)
    assert ret.returncode == 0
    ret = subprocess.run(command + ['--interface'])
    assert ret.returncode == 0


@pytest.mark.skipif(sys.platform == 'darwin', reason='No MPI support on macOS')
def test_create_mpi():
    """Ensure sdrun create works."""
    command = ['sdrun',
               'create',
               '-v',
               '-t', '2.50',
               '-s', '100',
               '--space-group', 'p2',
               '--lattice-lengths', '20', '24',
               '-o', 'test/output',
               '--hoomd-args', '"--mode=cpu"',
               'test/output/test_create.gsd',
               ]
    command = 'mpirun -np 4'.split(' ') + command
    ret = subprocess.run(command)
    assert ret.returncode == 0
    ret = subprocess.run(command + ['--interface'])
    assert ret.returncode == 0


def test_equil():
    """Ensure sdrun create works."""
    command = ['sdrun',
               'equil',
               '-v',
               '-t', '2.50',
               '-s', '100',
               '-o', 'test/output',
               '--hoomd-args', '"--mode=cpu"',
               'test/data/Trimer-13.50-3.00.gsd',
               'test/output/test_equil.gsd',
               ]
    ret = subprocess.run(command)
    assert ret.returncode == 0
    ret = subprocess.run(command + ['--equil-type', 'interface'])
    assert ret.returncode == 0


@pytest.mark.skipif(sys.platform == 'darwin', reason='No MPI support on macOS')
def test_equil_mpi():
    """Ensure sdrun create works."""
    command = ['sdrun',
               'equil',
               '-vvv',
               '-t', '2.50',
               '-s', '100',
               '-o', 'test/output',
               '--hoomd-args', '"--mode=cpu"',
               'test/data/Trimer-13.50-3.00.gsd',
               'test/output/test_equil.gsd',
               ]
    command = 'mpirun -np 4'.split(' ') + command
    ret = subprocess.run(command)
    assert ret.returncode == 0
    ret = subprocess.run(command + ['--equil-type', 'interface'])
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
    command = ['sdrun',
               'figure']
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(command, timeout=1)
