# -*- coding: utf-8 -*-

from sarpy.io.complex.naming.utils import get_pass_number

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

_orbits_per_day = 15.


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

    if not collector.lower().startswith('iceye'):
        return None

    crad = 'NI'
    cvehicle = collector[-2:].upper() if collector.lower().startswith('iceye-') else 'UN'
    pass_number = get_pass_number(cdate_mins, _orbits_per_day)

    return '{0:s}{1:s}{2:s}{3:s}{4:03d}'.format(cdate_str, crad, cvehicle, pass_number, product_number)
