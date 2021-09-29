"""
This module provide utilities for extracting a suggested name for a SICD.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
import os
import sys
import pkgutil
from importlib import import_module
from datetime import datetime

logger = logging.getLogger(__name__)

###########
# Module variables
_name_functions = []
_parsed_name_functions = False


def register_name_function(name_func):
    """
    Provide a new name function.

    Parameters
    ----------
    name_func : dict

    Returns
    -------
    None
    """

    if not callable(name_func):
        raise TypeError('name_func must be a callable')
    if name_func not in _name_functions:
        _name_functions.append(name_func)


def parse_name_functions():
    """
    Automatically find the viable name functions in the top-level modules.
    """

    global _parsed_name_functions
    if _parsed_name_functions:
        return
    _parsed_name_functions = True

    start_package = 'sarpy.io.complex.naming'
    module = import_module(start_package)
    for details in pkgutil.walk_packages(module.__path__, start_package+'.'):
        _, module_name, is_pkg = details
        if is_pkg:
            # don't bother checking for packages
            continue
        sub_module = import_module(module_name)
        if hasattr(sub_module, 'get_commercial_id'):
            register_name_function(sub_module.get_commercial_id)


def get_sicd_name(the_sicd, product_number=1):
    """
    Gets the suggested name.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
        The sicd structure.
    product_number : int
        The index of the product from the present file.

    Returns
    -------
    str
    """

    def get_commercial_id():
        commercial_id = None
        for entry in _name_functions:
            commercial_id = entry(collector, cdate_str, cdate_mins, product_number)
            if commercial_id is not None:
                break
        if commercial_id is None:
            return '{0:s}_{1:03d}'.format(the_sicd.CollectionInfo.CoreName, product_number)
        return commercial_id

    def get_vendor_id():
        _time_str = 'HHMMSS' if cdate is None else cdate.strftime('%H%M%S')
        _mode = '{}{}{}'.format(the_sicd.CollectionInfo.RadarMode.get_mode_abbreviation(),
                                the_sicd.Grid.get_resolution_abbreviation(),
                                the_sicd.SCPCOA.SideOfTrack)
        _coords = the_sicd.GeoData.SCP.get_image_center_abbreviation()
        _freq_band = the_sicd.RadarCollection.TxFrequency.get_band_abbreviation()
        _pol = '{}{}'.format(
            the_sicd.RadarCollection.get_polarization_abbreviation(),
            the_sicd.ImageFormation.get_polarization_abbreviation())
        return '_{}_{}_{}_001{}_{}_0101_SPY'.format(_time_str, _mode, _coords, _freq_band, _pol)

    # parse name function, if not already done
    parse_name_functions()

    # extract the common use variables
    if the_sicd.Timeline.CollectStart is None:
        cdate = None
        cdate_str = "DATE"
        cdate_mins = 0
    else:
        start_time = the_sicd.Timeline.CollectStart.astype('datetime64[s]')
        cdate = start_time.astype(datetime)
        cdate_str = cdate.strftime('%d%b%y')
        cdate_mins = cdate.hour * 60 + cdate.minute + cdate.second / 60.

    if the_sicd.CollectionInfo.CollectorName is None:
        collector = 'Unknown'
    else:
        collector = the_sicd.CollectionInfo.CollectorName.strip()

    # noinspection PyBroadException
    try:
        return get_commercial_id() + get_vendor_id()
    except Exception:
        logger.error('Failed to construct suggested name.')
        return None


def get_pass_number(minutes, orbits_per_day):
    """
    Gets appropriately formatted pass number string.

    Parameters
    ----------
    minutes : float
        Minutes elapsed in the day since midnight UTC.
    orbits_per_day : float
        The number of orbits per day, around 15 for vehicles in low earth orbit.

    Returns
    -------
    str
    """

    return '{0:02d}'.format(int(round(minutes*orbits_per_day/1440.)))
