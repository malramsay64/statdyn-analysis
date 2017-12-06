#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Create functions to colourize figures."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


HEX_VALUES_DARK = np.array([
    '#d08b98', '#cf8b97', '#cf8b96', '#cf8b96', '#cf8b95', '#cf8c94', '#cf8c93',
    '#cf8c92', '#ce8c91', '#ce8c90', '#ce8d8f', '#ce8d8e', '#ce8d8d', '#cd8d8c',
    '#cd8d8b', '#cd8e8a', '#cd8e89', '#cc8e89', '#cc8e88', '#cc8f87', '#cb8f86',
    '#cb8f85', '#cb8f84', '#ca8f83', '#ca9082', '#ca9081', '#c99080', '#c99080',
    '#c9917f', '#c8917e', '#c8917d', '#c7927c', '#c7927b', '#c6927a', '#c69279',
    '#c59379', '#c59378', '#c49377', '#c49376', '#c39475', '#c39475', '#c29474',
    '#c29573', '#c19572', '#c19572', '#c09571', '#c09670', '#bf966f', '#be966f',
    '#be976e', '#bd976d', '#bc976d', '#bc976c', '#bb986b', '#bb986b', '#ba986a',
    '#b9996a', '#b99969', '#b89968', '#b79a68', '#b69a67', '#b69a67', '#b59a66',
    '#b49b66', '#b49b66', '#b39b65', '#b29c65', '#b19c64', '#b09c64', '#b09c64',
    '#af9d63', '#ae9d63', '#ad9d63', '#ac9e63', '#ac9e62', '#ab9e62', '#aa9e62',
    '#a99f62', '#a89f62', '#a79f61', '#a7a061', '#a6a061', '#a5a061', '#a4a061',
    '#a3a161', '#a2a161', '#a1a161', '#a0a261', '#9fa261', '#9ea262', '#9da262',
    '#9ca362', '#9ba362', '#9aa362', '#9aa363', '#99a463', '#98a463', '#97a463',
    '#96a464', '#95a564', '#94a564', '#92a565', '#91a565', '#90a666', '#8fa666',
    '#8ea666', '#8da667', '#8ca767', '#8ba768', '#8aa768', '#89a769', '#88a76a',
    '#87a86a', '#86a86b', '#85a86b', '#83a86c', '#82a86d', '#81a96d', '#80a96e',
    '#7fa96f', '#7ea96f', '#7da970', '#7caa71', '#7aaa71', '#79aa72', '#78aa73',
    '#77aa74', '#76ab74', '#75ab75', '#73ab76', '#72ab77', '#71ab78', '#70ab78',
    '#6fab79', '#6eac7a', '#6cac7b', '#6bac7c', '#6aac7c', '#69ac7d', '#68ac7e',
    '#66ac7f', '#65ad80', '#64ad81', '#63ad82', '#62ad83', '#61ad83', '#5fad84',
    '#5ead85', '#5dad86', '#5cad87', '#5bad88', '#5aae89', '#58ae8a', '#57ae8b',
    '#56ae8c', '#55ae8c', '#54ae8d', '#53ae8e', '#52ae8f', '#51ae90', '#50ae91',
    '#4fae92', '#4eae93', '#4dae94', '#4cae95', '#4bae96', '#4aae96', '#49ae97',
    '#48ae98', '#48ae99', '#47ae9a', '#46ae9b', '#45ae9c', '#45ae9d', '#44ae9e',
    '#43ae9e', '#43ae9f', '#42aea0', '#42aea1', '#42aea2', '#41aea3', '#41aea4',
    '#41aea5', '#41aea5', '#40aea6', '#40aea7', '#40aea8', '#40aea9', '#40adaa',
    '#41adaa', '#41adab', '#41adac', '#41adad', '#42adae', '#42adae', '#43adaf',
    '#43adb0', '#44acb1', '#44acb2', '#45acb2', '#46acb3', '#46acb4', '#47acb5',
    '#48acb5', '#49abb6', '#4aabb7', '#4babb7', '#4cabb8', '#4dabb9', '#4eaab9',
    '#4faaba', '#50aabb', '#51aabb', '#52aabc', '#54a9bd', '#55a9bd', '#56a9be',
    '#57a9bf', '#59a8bf', '#5aa8c0', '#5ba8c0', '#5ca8c1', '#5ea7c1', '#5fa7c2',
    '#60a7c3', '#62a7c3', '#63a6c4', '#65a6c4', '#66a6c5', '#67a6c5', '#69a5c5',
    '#6aa5c6', '#6ba5c6', '#6da4c7', '#6ea4c7', '#70a4c8', '#71a3c8', '#72a3c8',
    '#74a3c9', '#75a3c9', '#77a2c9', '#78a2ca', '#79a2ca', '#7ba1ca', '#7ca1cb',
    '#7ea1cb', '#7fa0cb', '#80a0cb', '#82a0cc', '#839fcc', '#859fcc', '#869ecc',
    '#879ecd', '#899ecd', '#8a9dcd', '#8b9dcd', '#8d9dcd', '#8e9ccd', '#8f9ccd',
    '#909ccd', '#929bcd', '#939bce', '#949bce', '#969ace', '#979ace', '#9899ce',
    '#9999ce', '#9a99ce', '#9c98ce', '#9d98ce', '#9e98cd', '#9f97cd', '#a097cd',
    '#a297cd', '#a396cd', '#a496cd', '#a596cd', '#a695cd', '#a795cc', '#a894cc',
    '#a994cc', '#aa94cc', '#ab93cc', '#ac93cb', '#ad93cb', '#ae92cb', '#af92cb',
    '#b092ca', '#b191ca', '#b291ca', '#b391c9', '#b491c9', '#b590c9', '#b690c8',
    '#b790c8', '#b88fc7', '#b88fc7', '#b98fc7', '#ba8fc6', '#bb8ec6', '#bc8ec5',
    '#bc8ec5', '#bd8ec4', '#be8dc4', '#bf8dc3', '#bf8dc3', '#c08dc2', '#c18cc2',
    '#c18cc1', '#c28cc0', '#c38cc0', '#c38cbf', '#c48bbf', '#c48bbe', '#c58bbd',
    '#c68bbd', '#c68bbc', '#c78bbc', '#c78abb', '#c88aba', '#c88ab9', '#c98ab9',
    '#c98ab8', '#c98ab7', '#ca8ab7', '#ca8ab6', '#cb8ab5', '#cb89b4', '#cb89b4',
    '#cc89b3', '#cc89b2', '#cc89b1', '#cd89b1', '#cd89b0', '#cd89af', '#ce89ae',
    '#ce89ad', '#ce89ad', '#ce89ac', '#ce89ab', '#cf89aa', '#cf89a9', '#cf89a8',
    '#cf89a8', '#cf89a7', '#cf89a6', '#cf89a5', '#d089a4', '#d08aa3', '#d08aa2',
    '#d08aa1', '#d08aa1', '#d08aa0', '#d08a9f', '#d08a9e', '#d08a9d', '#d08a9c',
    '#d08a9b', '#d08b9a', '#d08b99',
])

HEX_VALUES_LIGHT = np.array([
    '#f9c8d1', '#f9c8d0', '#f9c8cf', '#f9c8cf', '#f9c8ce', '#f9c8ce', '#f9c8cd',
    '#f9c9cc', '#f8c9cc', '#f8c9cb', '#f8c9ca', '#f8c9ca', '#f8c9c9', '#f8c9c9',
    '#f8c9c8', '#f7c9c7', '#f7cac7', '#f7cac6', '#f7cac6', '#f7cac5', '#f6cac4',
    '#f6cac4', '#f6cac3', '#f6cbc3', '#f5cbc2', '#f5cbc2', '#f5cbc1', '#f5cbc1',
    '#f4cbc0', '#f4ccbf', '#f4ccbf', '#f3ccbe', '#f3ccbe', '#f3ccbd', '#f2ccbd',
    '#f2cdbc', '#f2cdbc', '#f1cdbb', '#f1cdbb', '#f0cdba', '#f0ceba', '#f0ceb9',
    '#efceb9', '#efceb9', '#eeceb8', '#eecfb8', '#eecfb7', '#edcfb7', '#edcfb6',
    '#eccfb6', '#eccfb6', '#ebd0b5', '#ebd0b5', '#ead0b5', '#ead0b4', '#e9d0b4',
    '#e9d1b3', '#e8d1b3', '#e8d1b3', '#e7d1b3', '#e7d2b2', '#e6d2b2', '#e6d2b2',
    '#e5d2b1', '#e4d2b1', '#e4d3b1', '#e3d3b1', '#e3d3b1', '#e2d3b0', '#e1d3b0',
    '#e1d4b0', '#e0d4b0', '#e0d4b0', '#dfd4b0', '#ded4af', '#ded5af', '#ddd5af',
    '#ddd5af', '#dcd5af', '#dbd5af', '#dbd6af', '#dad6af', '#d9d6af', '#d9d6af',
    '#d8d6af', '#d7d7af', '#d7d7af', '#d6d7af', '#d5d7af', '#d5d7af', '#d4d8af',
    '#d3d8af', '#d3d8af', '#d2d8af', '#d1d8b0', '#d1d9b0', '#d0d9b0', '#cfd9b0',
    '#ced9b0', '#ced9b0', '#cdd9b0', '#ccdab1', '#ccdab1', '#cbdab1', '#cadab1',
    '#cadab2', '#c9dab2', '#c8dbb2', '#c7dbb2', '#c7dbb3', '#c6dbb3', '#c5dbb3',
    '#c5dbb4', '#c4dcb4', '#c3dcb4', '#c3dcb5', '#c2dcb5', '#c1dcb5', '#c0dcb6',
    '#c0ddb6', '#bfddb7', '#beddb7', '#beddb7', '#bdddb8', '#bcddb8', '#bcddb9',
    '#bbddb9', '#badeba', '#badeba', '#b9deba', '#b8debb', '#b8debb', '#b7debc',
    '#b6debc', '#b6debd', '#b5debd', '#b5dfbe', '#b4dfbe', '#b3dfbf', '#b3dfc0',
    '#b2dfc0', '#b2dfc1', '#b1dfc1', '#b0dfc2', '#b0dfc2', '#afdfc3', '#afdfc3',
    '#aedfc4', '#aedfc5', '#ade0c5', '#ade0c6', '#ace0c6', '#ace0c7', '#abe0c7',
    '#abe0c8', '#aae0c9', '#aae0c9', '#aae0ca', '#a9e0ca', '#a9e0cb', '#a8e0cc',
    '#a8e0cc', '#a8e0cd', '#a7e0cd', '#a7e0ce', '#a7e0cf', '#a6e0cf', '#a6e0d0',
    '#a6e0d1', '#a6e0d1', '#a5e0d2', '#a5e0d2', '#a5e0d3', '#a5e0d4', '#a5e0d4',
    '#a5e0d5', '#a4e0d5', '#a4e0d6', '#a4e0d7', '#a4e0d7', '#a4e0d8', '#a4e0d8',
    '#a4e0d9', '#a4e0da', '#a4dfda', '#a4dfdb', '#a4dfdb', '#a4dfdc', '#a4dfdc',
    '#a4dfdd', '#a4dfde', '#a5dfde', '#a5dfdf', '#a5dfdf', '#a5dfe0', '#a5dfe0',
    '#a6dfe1', '#a6dee2', '#a6dee2', '#a6dee3', '#a7dee3', '#a7dee4', '#a7dee4',
    '#a8dee5', '#a8dee5', '#a8dde6', '#a9dde6', '#a9dde7', '#aadde7', '#aadde8',
    '#aadde8', '#abdde9', '#abdce9', '#acdcea', '#acdcea', '#addcea', '#addceb',
    '#aedceb', '#aedbec', '#afdbec', '#b0dbec', '#b0dbed', '#b1dbed', '#b1dbee',
    '#b2daee', '#b3daee', '#b3daef', '#b4daef', '#b5daef', '#b5d9f0', '#b6d9f0',
    '#b7d9f0', '#b7d9f1', '#b8d9f1', '#b9d8f1', '#bad8f2', '#bad8f2', '#bbd8f2',
    '#bcd8f2', '#bcd7f3', '#bdd7f3', '#bed7f3', '#bfd7f3', '#bfd7f4', '#c0d6f4',
    '#c1d6f4', '#c2d6f4', '#c3d6f4', '#c3d5f4', '#c4d5f5', '#c5d5f5', '#c6d5f5',
    '#c6d5f5', '#c7d4f5', '#c8d4f5', '#c9d4f5', '#cad4f5', '#cad3f6', '#cbd3f6',
    '#ccd3f6', '#cdd3f6', '#ced2f6', '#ced2f6', '#cfd2f6', '#d0d2f6', '#d1d2f6',
    '#d1d1f6', '#d2d1f6', '#d3d1f6', '#d4d1f6', '#d4d0f6', '#d5d0f6', '#d6d0f6',
    '#d7d0f5', '#d8d0f5', '#d8cff5', '#d9cff5', '#dacff5', '#dacff5', '#dbcef5',
    '#dccef5', '#ddcef4', '#ddcef4', '#decef4', '#dfcdf4', '#dfcdf4', '#e0cdf4',
    '#e1cdf3', '#e1cdf3', '#e2ccf3', '#e3ccf3', '#e3ccf2', '#e4ccf2', '#e5ccf2',
    '#e5cbf2', '#e6cbf1', '#e6cbf1', '#e7cbf1', '#e8cbf0', '#e8cbf0', '#e9caf0',
    '#e9caef', '#eacaef', '#eacaef', '#ebcaee', '#eccaee', '#eccaee', '#edc9ed',
    '#edc9ed', '#eec9ec', '#eec9ec', '#eec9ec', '#efc9eb', '#efc9eb', '#f0c9ea',
    '#f0c8ea', '#f1c8e9', '#f1c8e9', '#f2c8e8', '#f2c8e8', '#f2c8e7', '#f3c8e7',
    '#f3c8e7', '#f3c8e6', '#f4c8e5', '#f4c8e5', '#f4c8e4', '#f5c7e4', '#f5c7e3',
    '#f5c7e3', '#f6c7e2', '#f6c7e2', '#f6c7e1', '#f6c7e1', '#f7c7e0', '#f7c7e0',
    '#f7c7df', '#f7c7de', '#f7c7de', '#f8c7dd', '#f8c7dd', '#f8c7dc', '#f8c7dc',
    '#f8c7db', '#f8c7da', '#f9c7da', '#f9c7d9', '#f9c7d9', '#f9c7d8', '#f9c7d7',
    '#f9c7d7', '#f9c7d6', '#f9c7d6', '#f9c7d5', '#f9c7d4', '#f9c8d4', '#f9c8d3',
    '#f9c8d2', '#f9c8d2', '#f9c8d1',
])


def colour_orientation(orientations, light_colours=False):
    """Get a colour from an orientation."""
    index = np.floor(orientations / np.pi * 180).astype(int) + 180
    index %= 360
    if light_colours:
        return HEX_VALUES_LIGHT[index]
    return HEX_VALUES_DARK[index]
