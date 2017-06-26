#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Compute motion from a trajectory."""

import argparse
from pathlib import Path

from statdyn.datagen.trajectory import compute_motion


def main():
    """Run from command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='File to process')
    parser.add_argument(
        '-o',
        '--outdir',
        default=None,
        help='''Directory to output files to. Default is to use the same
        directory as the input file'''
    )
    args = parser.parse_args()
    compute_motion(Path(args.filename), Path(args.outdir))


if __name__ == "__main__":
    main()
