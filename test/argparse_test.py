#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the parsing of arguments gives the correct results."""

import pytest


def test_verbose():
    pass


def test_version():
    pass


@pytest.mark.parametrize(
    "extras",
    [
        ["test"],
        ["--argument"],
        ["--argument", "with_value"],
        ["-a", "--argument", "value", "123"],
    ],
)
def test_bokeh(extras):
    pass
