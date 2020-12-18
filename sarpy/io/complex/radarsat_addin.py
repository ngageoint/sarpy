# -*- coding: utf-8 -*-
"""
Specific functionality add-in. **Note that this is NOT part of the standard repository.**
"""

import logging

__author__ = "Thomas McCullough"
__note__ = "Not intended for public dissemination"
__classification__ = "FOUO"


def extract_radarsat_sec(nitf, class_str):
    """
    Populate the security tags and extract the classification string for the SICD.

    Parameters
    ----------
    nitf : dict
    class_str : str

    Returns
    -------
    str
    """

    if 'UNCLASS' in class_str:
        classification = 'UNCLASSIFIED'
    elif class_str == 'CAN SECRET':
        classification = '//CAN SECRET//REL TO CAN, FVEY'
        nitf['Security'] = {'CLAS': 'S', 'CLSY': 'CA', 'CODE': ''}
    else:
        logging.critical('Unsure how to handle RCM classification string {}, so we are '
                         'passing it straight through.'.format(class_str))
        classification = class_str
    return classification
