__classification__ = 'UNCLASSIFIED'
__author__ = "Thomas McCullough"

import os


def get_schema_path(the_urn):
    """
    Gets the path to the proper schema file for the given urn.

    Parameters
    ----------
    the_urn : str

    Returns
    -------
    str
    """

    the_directory = os.path.split(__file__)[0]
    if the_urn == 'urn:SIDD:1.0.0':
        return os.path.join(
            the_directory, 'version1', 'SIDD_schema_V1.0.0_2011_08_31.xsd')
    elif the_urn == 'urn:SIDD:2.0.0':
        return os.path.join(
            the_directory, 'version2', 'SIDD_schema_V2.0.0_2019_05_31.xsd')
    else:
        raise ValueError('Got unrecognized urn {}'.format(the_urn))
