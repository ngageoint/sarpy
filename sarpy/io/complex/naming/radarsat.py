# -*- coding: utf-8 -*-

from sarpy.io.complex.naming.utils import get_pass_number

__classification__ = "FOUO"
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
    else:
        return None

    pass_number = get_pass_number(cdate_mins, _orbits_per_day)
    return '{0:s}{1:s}{2:s}{3:s}{4:03d}'.format(cdate_str, crad, cvehicle, pass_number, product_number)
