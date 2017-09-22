#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Options for sdrun."""

import logging

logger = logging.getLogger(__name__)


def _verbosity(ctx, param, count):
    root_logger = logging.getLogger('statdyn')
    if not count or ctx.resilient_parsing:
        logging.basicConfig(level=logging.WARNING)
        root_logger.setLevel(logging.WARNING)
        return
    if count == 1:
        logging.basicConfig(level=logging.INFO)
        root_logger.setLevel(logging.INFO)
        logger.info('Set log level to INFO')
    if count > 1:
        logging.basicConfig(level=logging.DEBUG)
        root_logger.setLevel(logging.DEBUG)
        logger.info('Setting log level to DEBUG')
    root_logger.debug('Logging set for root')

