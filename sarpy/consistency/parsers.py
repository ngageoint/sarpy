# -*- coding: utf-8 -*-
#
# Copyright 2020-2021 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#

__classification__ = "UNCLASSIFIED"
__author__ = "Nathan Bombaci, Valkyrie"


from typing import List, Callable
import numpy as np


def parse_text(elem):
    """
    Reverse of `xml.make_elem` by converting an element's text string to
    an int, float, bool, or str, as appropriate.

    Parameters
    ----------
    elem: lxml.etree.ElementTree.Element
        Element to convert to the most restrictive python type possible.

    Returns
    -------
    val: int|float|bool|str
        Converted value.
    """

    for converter in (int, float, parse_bool_text, str):
        try:
            val = converter(elem.text)
            break
        except ValueError:
            continue
    return val


def parse_bool_text(text):
    """
    Gets a boolean from a string.

    Parameters
    ----------
    text: str
        One of `'true', '1', 'false', '0'`.

    Returns
    -------
    val: bool
        Boolean value converted from `text`.

    Raises
    ------
    ValueError
        The text string is not either ``'true'`` or ``'false'``.
    """

    text = text.lower()
    if text in ['true', '1']:
        return True
    if text in ['false', '0']:
        return False
    raise ValueError("Cannot parse bool from {}".format(text))


def parse_bool(elem):
    """
    Gets a boolean from an element.

    Parameters
    ----------
    elem : lxml.etree.ElementTree.Element
        Element to convert.

    Returns
    -------
    val : bool
        Boolean value of the `elem`'s text.
    """

    return parse_bool_text(elem.text)


def parse_sequence(node, keys, conversion=parse_text):
    """
    Reverse of `sequence_node`.

    Parameters
    ----------
    node : lxml.etree.ElementTree.Element
        Element containing a sequence node.
    keys : List
        List of element names to parse.
    conversion : Callable
        Conversion function. (Default: `parse_text`)

    Returns
    ------
    List
        List of parsed values, one for each element of `keys`.
    """

    return [conversion(node.find('./' + key)) for key in keys]


def parse_xyz(node):
    """
    Parse a node with ``'X'``, ``'Y'``, and ``'Z'`` children

    Parameters
    ----------
    node : lxml.etree.ElementTree.Element
        Element containing an XYZ sequence node.

    Returns
    -------
    List
        List [X, Y, Z]. Parsed values.
    """

    return parse_sequence(node, ['X', 'Y', 'Z'], lambda x: float(x.text))


def parse_xy(node):
    """
    Parse a node with ``'X'`` and ``'Y'`` children.

    Parameters
    ----------
    node: lxml.etree.ElementTree.Element
        Element containing an XY sequence node.

    Returns
    -------
    List
        List [X, Y]. Parsed values.
    """

    return parse_sequence(node, ['X', 'Y'], lambda x: float(x.text))


def parse_ll(node):
    """
    Parse a node with ``'Lat'`` and ``'Lon'`` children.

    Parameters
    ----------
    node: lxml.etree.ElementTree.Element
        Element containing a Lat/Lon sequence node.

    Returns
    -------
    List
        List [Lon, Lat]. Parsed values as radians.
    """

    return np.radians(float(node.findtext('Lon'))), np.radians(float(node.findtext('Lat')))
