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
    if the_urn == 'urn:SICD:0.3.1':
        return os.path.join(the_directory, 'SICD_schema_V0_3_1_2009_03_17.xsd')
    elif the_urn == 'urn:SICD:0.4.0':
        return os.path.join(the_directory, 'SICD_schema_V0_4_0_2010_02_12.xsd')
    elif the_urn == 'urn:SICD:0.4.1':
        return os.path.join(the_directory, 'SICD_schema_V0_4_1_2010_07_15.xsd')
    elif the_urn == 'urn:SICD:0.5.0':
        return os.path.join(the_directory, 'SICD_schema_V0_5_0_2011_01_12.xsd')
    elif the_urn == 'urn:SICD:1.0.0':
        return os.path.join(the_directory, 'SICD_schema_V1_0_0_2011_08_31.xsd')
    elif the_urn == 'urn:SICD:1.0.1':
        return os.path.join(the_directory, 'SICD_schema_V1_0_1_2013_02_25.xsd')
    elif the_urn == 'urn:SICD:1.1.0':
        return os.path.join(the_directory, 'SICD_schema_V1_1_0_2014_07_08.xsd')
    elif the_urn == 'urn:SICD:1.2.0':
        return os.path.join(the_directory, 'SICD_schema_V1_2_0_2016_06_30.xsd')
    elif the_urn == 'urn:SICD:1.2.1':
        return os.path.join(the_directory, 'SICD_schema_V1_2_1_2018_12_13.xsd')
    else:
        raise ValueError('Got unrecognized urn {}'.format(the_urn))
