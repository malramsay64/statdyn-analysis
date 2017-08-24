#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Generate summary statistics for dynamics quantities.

This provides the functions required to generate summary statistics for many
different dynamics quantities.
"""

from pathlib import Path

import click
import pandas

from . import collate, trajectory
from ..sdrun import options


@click.command()
@options.arg_infile
def motion(infile):
    """Compute the motion of each molecule in every frame."""
    trajectory.compute_motion(Path(infile))


@click.command()
@options.arg_infile
@options.arg_outfile
def dynamics(infile, outfile):
    """Compute dynamic quantities."""
    infile = Path(infile)
    outfile = Path(outfile)
    src = pandas.read_hdf(infile)
    data = pandas.DataFrame({
        'gamma': collate.allgroups(src, func=collate.gamma),
        'overlap': collate.allgroups(src, func=collate.overlap),
        'alpha': collate.allgroups(src, func=collate.alpha),
        'spearman': collate.allgroups(src, func=collate.spearman_rank),
    })
    data.to_hdf(str(outfile), 'temperature')
