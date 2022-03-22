"""
This package contains the AFRL RDE schema
"""

__classification__ = 'UNCLASSIFIED'

import pkg_resources


def get_schema_path(version='1.0.0'):
    """
    Location of AFRL/RDE schema file.

    Returns
    -------
    str
        The path to the ARFL/RDE schema.
    """

    if version == '1.0.0':
        return pkg_resources.resource_filename(
            'sarpy.annotation.afrl_rde_schema', 'afrl_rde_schema_v1.0.0_2022-02-15.xsd')
    else:
        raise ValueError('Got unrecognized version {}'.format(version))
