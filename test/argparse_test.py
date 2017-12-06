#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the parsing of arguments gives the correct results."""

import pytest

from sdanalysis.main import create_parser

parser = create_parser()

FUNCS = [
    ('figure', []),
    ('comp_dynamics', ['infile']),
]


@pytest.mark.parametrize('func, extras', FUNCS)
def test_verbose(func, extras):
    args = parser.parse_args([func, '-v'] + extras)
    assert args.verbose == 1
    args = parser.parse_args([func, '--verbose'] + extras)
    assert args.verbose == 1
    args = parser.parse_args([func] + ['-v']*3 + extras)
    assert args.verbose == 3


@pytest.mark.parametrize('func, extras', FUNCS)
def test_version(func, extras):
    with pytest.raises(SystemExit) as e:
        parser.parse_args(['--version'])
        assert e == 0
    with pytest.raises(SystemExit) as e:
        parser.parse_args([func, '--version'])
        assert e == 0
    with pytest.raises(SystemExit) as e:
        parser.parse_args(['--version', func])
        assert e == 0


@pytest.mark.parametrize('extras', [
    ['test'],
    ['--argument'],
    ['--argument', 'with_value'],
    ['-a', '--argument', 'value', '123']
])
def test_bokeh(extras):
    args = parser.parse_args(['figure', '--'] + extras)
    assert args.bokeh == extras
    args = parser.parse_args(['figure', '"{}"'.format(' '.join(extras))])
    assert args.bokeh == ['"{}"'.format(' '.join(extras))]
