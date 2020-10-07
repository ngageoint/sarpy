# -*- coding: utf-8 -*-

import re

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

_orbits_per_day = 14.292


def get_commercial_id(collector, cdate_str, cdate_mins, product_number):
    """
    Gets the commercial id, if appropriate.

    Parameters
    ----------
    collector : str
    cdate : datetime.datetime
    cdate_str : str
    cdate_mins : float
    product_number : int

    Returns
    -------
    None|str
    """

    if collector.upper() in ('RADARSAT-1', 'RADARSAT-2'):
        crad = 'RS'
        cvehicle = '0'+collector[-1]
    elif collector.startswith('RCM'):
        crad = 'RC'
        cvehicle = '{0:02d}'.format(int(re.sub('-', '', collector[3:])))
    else:
        return None

    pass_number = '{0:02d}'.format(int(round(cdate_mins * _orbits_per_day / 1440.)))
    return '{0:s}{1:s}{2:s}{3:s}{4:03d}'.format(cdate_str, crad, cvehicle, pass_number, product_number)
