# -*- coding: utf-8 -*-
"""
This module provide utilities for extracting a suggested name for a SICD.
"""

import logging
import os
import sys
import pkgutil
from importlib import import_module
from datetime import datetime

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


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

    Returns
    -------

    """

    global _parsed_name_functions
    if _parsed_name_functions:
        return
    _parsed_name_functions = True

    def check_module(mod_name):
        # import the module
        import_module(mod_name)
        # fetch the module from the modules dict
        module = sys.modules[mod_name]
        # see if it has an is_a function, if so, register it
        if hasattr(module, 'get_commercial_id'):
            register_name_function(module.get_commercial_id)

        # walk down any subpackages
        path, fil = os.path.split(module.__file__)
        if not fil.startswith('__init__.py'):
            # there are no subpackages
            return
        for sub_module in pkgutil.walk_packages([path, ]):
            _, sub_module_name, _ = sub_module
            sub_name = "{}.{}".format(mod_name, sub_module_name)
            check_module(sub_name)

    check_module('sarpy.io.complex.naming')


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
        commericial_id = None
        for entry in _name_functions:
            commericial_id = entry(collector, cdate_str, cdate_mins, product_number)
            if commericial_id is not None:
                break
        if commericial_id is None:
            return '{0:s}_{1:03d}'.format(the_sicd.CollectionInfo.CoreName, product_number)
        return commericial_id

    def get_vendor_id():
        _time_str = cdate.strftime('%H%M%S')
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
    cdate = the_sicd.Timeline.CollectStart.astype(datetime)
    cdate_str = cdate.strftime('%d%b%y')
    cdate_mins = cdate.hour * 60 + cdate.minute + cdate.second / 60.
    collector = the_sicd.CollectionInfo.CollectorName.strip()

    try:
        return get_commercial_id() + get_vendor_id()
    except AttributeError:
        logging.error('Failed to construct suggested name.')
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
