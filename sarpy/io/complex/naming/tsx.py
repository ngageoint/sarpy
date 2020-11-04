# -*- coding: utf-8 -*-

from sarpy.io.complex.naming.utils import get_pass_number

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

_orbits_per_day = 15.182


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

    if not (collector.lower().startswith('terra') or collector.lower().startswith('tsx')):
        return None

    crad = 'TS'
    vehicle_parts = collector.split('-')
    if len(vehicle_parts) > 1:
        cvehicle = vehicle_parts[-1]
        if len(cvehicle) > 2:
            cvehicle = cvehicle[:2]
        elif len(cvehicle) < 2:
            cvehicle = '0'*(2-len(cvehicle)) + cvehicle
    else:
        cvehicle = 'UN'
    pass_number = get_pass_number(cdate_mins, _orbits_per_day)
    return '{0:s}{1:s}{2:s}{3:s}{4:03d}'.format(cdate_str, crad, cvehicle, pass_number, product_number)
