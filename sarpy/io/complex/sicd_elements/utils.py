"""
Common use sicd_elements methods.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import numpy
from sarpy.io.general.utils import get_seconds


def _get_center_frequency(RadarCollection, ImageFormation):
    """
    Helper method.

    Parameters
    ----------
    RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
    ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType

    Returns
    -------
    None|float
        The center processed frequency, in the event that RadarCollection.RefFreqIndex is `None` or `0`.
    """

    if RadarCollection is None or RadarCollection.RefFreqIndex is None or RadarCollection.RefFreqIndex == 0:
        return None
    if ImageFormation is None or ImageFormation.TxFrequencyProc is None or \
            ImageFormation.TxFrequencyProc.MinProc is None or ImageFormation.TxFrequencyProc.MaxProc is None:
        return None
    return 0.5 * (ImageFormation.TxFrequencyProc.MinProc + ImageFormation.TxFrequencyProc.MaxProc)


def is_polstring_version1(str_in):
    """
    Is the polarization string compatible for SCD version 1.1?

    Parameters
    ----------
    str_in : None|str
        The tx/rcv polarization string.

    Returns
    -------
    bool
    """

    if str_in is None or str_in in ['OTHER', 'UNKNOWN']:
        return True

    parts = str_in.split(':')
    if len(parts) != 2:
        return False

    part1, part2 = parts
    if (part1 in ['V', 'H'] and part2 in ['RHC', 'LHC']) or (part2 in ['V', 'H'] and part1 in ['RHC', 'LHC']):
        return False
    return True


################
# SICD comparsion and matching methods

def is_same_size(sicd1, sicd2):
    """
    Are the two SICD structures the same size in pixels?

    Parameters
    ----------
    sicd1 : sarpy.io.complex.sicd_elements.SICD.SICDType
    sicd2 : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if sicd1 is sicd2:
        return True

    try:
        return (sicd1.ImageData.NumRows == sicd2.ImageData.NumRows) and \
               (sicd1.ImageData.NumCols == sicd2.ImageData.NumCols)
    except AttributeError:
        return False


def is_same_sensor(sicd1, sicd2):
    """
    Are the two SICD structures from the same sensor?

    Parameters
    ----------
    sicd1 : sarpy.io.complex.sicd_elements.SICD.SICDType
    sicd2 : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if sicd1 is sicd2:
        return True

    try:
        return sicd1.CollectionInfo.CollectorName == sicd2.CollectionInfo.CollectorName
    except AttributeError:
        return False


def is_same_start_time(sicd1, sicd2):
    """
    Do the two SICD structures have the same start time with millisecond resolution?

    Parameters
    ----------
    sicd1 : sarpy.io.complex.sicd_elements.SICD.SICDType
    sicd2 : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if sicd1 is sicd2:
        return True

    try:
        return abs(get_seconds(sicd1.Timeline.CollectStart, sicd2.Timeline.CollectStart, precision='ms')) < 2e-3
    except AttributeError:
        return False


def is_same_duration(sicd1, sicd2):
    """
    Do the two SICD structures have the same duration, with millisecond resolution?

    Parameters
    ----------
    sicd1 : sarpy.io.complex.sicd_elements.SICD.SICDType
    sicd2 : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if sicd1 is sicd2:
        return True

    try:
        return abs(sicd1.Timeline.CollectDuration - sicd2.Timeline.CollectDuration) < 2e-3
    except AttributeError:
        return False


def is_same_band(sicd1, sicd2):
    """
    Are the two SICD structures the same band?

    Parameters
    ----------
    sicd1 : sarpy.io.complex.sicd_elements.SICD.SICDType
    sicd2 : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if sicd1 is sicd2:
        return True

    try:
        return abs(sicd1.Grid.Row.KCtr - sicd2.Grid.Row.KCtr) <= 1./(sicd1.Grid.Row.SS*sicd1.ImageData.NumRows)
    except AttributeError:
        return False


def is_same_scp(sicd1, sicd2):
    """
    Do the two SICD structures share the same SCP, with resolution of one meter
    in each ECF coordinate?

    Parameters
    ----------
    sicd1 : sarpy.io.complex.sicd_elements.SICD.SICDType
    sicd2 : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if sicd1 is sicd2:
        return True

    try:
        ecf1 = sicd1.GeoData.SCP.ECF.get_array()
        ecf2 = sicd2.GeoData.SCP.ECF.get_array()
        return numpy.all(numpy.abs(ecf1 - ecf2) < 1)
    except AttributeError:
        return False


def is_general_match(sicd1, sicd2):
    """
    Do the two SICD structures seem to form a basic match? This necessarily
    establishes and equivalence relation between sicds.

    Parameters
    ----------
    sicd1 : sarpy.io.complex.sicd_elements.SICD.SICDType
    sicd2 : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if sicd1 is sicd2:
        return True

    return is_same_size(sicd1, sicd2) and is_same_sensor(sicd1, sicd2) and \
           is_same_start_time(sicd1, sicd2) and is_same_duration(sicd1, sicd2) and \
           is_same_band(sicd1, sicd2) and is_same_scp(sicd1, sicd2)
