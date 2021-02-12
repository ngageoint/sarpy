# -*- coding: utf-8 -*-
"""
This package contains the CPHD schema
"""

__classification__ = 'UNCLASSIFIED'

import pkg_resources


def location():
    """
    Location of CPHD schema file.

    Returns
    -------
    str
        The path to the CPHD schema.
    """

    return pkg_resources.resource_filename('sarpy.io.phase_history.cphd_schema', 'CPHD_schema_V1.0.1_2018_05_21.xsd')
