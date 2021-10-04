"""
This package contains the CRSD schema
"""

__classification__ = 'UNCLASSIFIED'

import pkg_resources


def get_schema_path(version='1.0.0'):
    """
    Location of CRSD schema file.

    Parameters
    ----------
    version : str

    Returns
    -------
    str
        The path to the CRSD schema.
    """

    if version == '1.0.0':
        return pkg_resources.resource_filename('sarpy.io.received.crsd_schema', 'CRSD_schema_V1.0.0_2021_06_12.xsd')
    else:
        raise ValueError('Got unrecognized version {}'.format(version))
